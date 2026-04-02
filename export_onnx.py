"""Export model_700.pt actor network to ONNX for deployment on Raspberry Pi.

Run with the unitree_rl conda environment:
  conda run -n unitree_rl python3 export_onnx.py

Output: policy.onnx in the same directory as this script.

Observation vector (45 elements, flat-terrain model):
  [0:3]   base_ang_vel       - IMU gyro (rad/s) in body frame
  [3:6]   projected_gravity  - gravity vector in body frame, normalized [-1..1]
  [6:9]   command            - [vx, vy, wz] velocity command
  [9:21]  joint_pos          - revolute1..12 positions (rad), relative to default=0
  [21:33] joint_vel          - revolute1..12 velocities (rad/s)
  [33:45] last_action        - previous raw network output (before * 0.25 scaling)

Output (12 elements): raw action means.
  target_joint_pos = output * 0.25  (scale factor from training)
  then clip to [-3.14, 3.14] per joint
"""

import torch
import torch.nn as nn
from pathlib import Path

OBS_DIM = 45
ACTION_DIM = 12


class PolicyONNX(nn.Module):
    """Actor network with baked-in observation normalization.

    Inference pipeline:
      raw_obs -> normalize -> MLP (ELU) -> raw_action
    """

    def __init__(self, actor_sd: dict):
        super().__init__()

        # Observation normalizer statistics (shape: [1, 45])
        self.register_buffer("norm_mean", actor_sd["obs_normalizer._mean"].float())
        # Use _std (already computed as sqrt(var + eps) during training)
        self.register_buffer("norm_std", actor_sd["obs_normalizer._std"].float())

        # MLP: Linear -> ELU -> Linear -> ELU -> Linear -> ELU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(OBS_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        )

        # Load weights from checkpoint state dict
        mlp_sd = {
            "0.weight": actor_sd["mlp.0.weight"],
            "0.bias":   actor_sd["mlp.0.bias"],
            "2.weight": actor_sd["mlp.2.weight"],
            "2.bias":   actor_sd["mlp.2.bias"],
            "4.weight": actor_sd["mlp.4.weight"],
            "4.bias":   actor_sd["mlp.4.bias"],
            "6.weight": actor_sd["mlp.6.weight"],
            "6.bias":   actor_sd["mlp.6.bias"],
        }
        self.mlp.load_state_dict(mlp_sd)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: [batch, 45]
        # Clamp std to avoid div-by-zero (matches training normalizer)
        std = self.norm_std.clamp(min=1e-8)
        norm_obs = (obs - self.norm_mean) / std
        return self.mlp(norm_obs)


def main():
    workspace = Path(__file__).parent
    checkpoint = workspace / "model_700.pt"
    output = workspace / "policy.onnx"

    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    actor_sd = ckpt["actor_state_dict"]

    policy = PolicyONNX(actor_sd)
    policy.eval()

    dummy_input = torch.zeros(1, OBS_DIM)

    print(f"Exporting to: {output}")
    torch.onnx.export(
        policy,
        dummy_input,
        str(output),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=11,
    )

    # Quick sanity check
    import onnx
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
              for i in model.graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
               for o in model.graph.output]
    print(f"ONNX model OK — inputs: {inputs}, outputs: {outputs}")

    # Verify numerically
    import numpy as np
    obs_np = np.zeros((1, OBS_DIM), dtype=np.float32)
    with torch.no_grad():
        pt_out = policy(torch.from_numpy(obs_np)).numpy()

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"obs": obs_np})[0]
        max_diff = float(np.abs(pt_out - ort_out).max())
        print(f"PyTorch vs ONNX max diff: {max_diff:.6e}")
    except ImportError:
        print("onnxruntime not installed — skipping numerical check")

    print("Done.")


if __name__ == "__main__":
    main()
