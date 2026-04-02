"""kutta_rl_controller — ROS 2 node.

Loads the trained policy (ONNX), reads sensor topics, and publishes joint
position commands at 50 Hz — exactly mirroring what the rl_mjlab play.py
inference loop does inside MuJoCo.

Subscribed topics
-----------------
/joint_states    sensor_msgs/JointState   12 joint positions + velocities
/imu/data        sensor_msgs/Imu          angular velocity + orientation
/cmd_vel         geometry_msgs/Twist      desired [vx, vy, wz]

Published topics
----------------
/kutta/joint_commands  std_msgs/Float64MultiArray  12 target joint positions (rad)

Observation vector (45 elements — matches actor network input):
  [0:3]   base_ang_vel       IMU angular velocity in body frame  (rad/s)
  [3:6]   projected_gravity  gravity unit-vector in body frame
  [6:9]   command            [vx, vy, wz] from /cmd_vel
  [9:21]  joint_pos          revolute1..12, relative to default=0  (rad)
  [21:33] joint_vel          revolute1..12  (rad/s)
  [33:45] last_action        previous raw network output (before ×0.25)

Action pipeline (matches JointPositionActionCfg in velocity_env_cfg.py):
  raw_action = network_output          (12 values)
  target_pos = raw_action × 0.25      (scale=0.25, default_offset=0)
  target_pos = clip(target_pos, -3.14, 3.14)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

# Joint order must match training (revolute1..12 = FL-hip, FL-thigh, FL-knee,
# RL-hip, RL-thigh, RL-knee, FR-hip, FR-thigh, FR-knee, RR-hip, RR-thigh, RR-knee)
JOINT_NAMES = [f"revolute{i}" for i in range(1, 13)]

OBS_DIM = 45
ACTION_DIM = 12
ACTION_SCALE = 0.25          # From JointPositionActionCfg(scale=0.25)
JOINT_LIMIT = 3.14           # actuator ctrlrange in kutta.xml
CTRL_HZ = 50.0               # decimation=4, sim_dt=0.005 → policy_dt=0.02 s


def _quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the inverse of quaternion q = [w, x, y, z].

    If q represents rotation from body frame to world frame, this returns v
    expressed in body frame.  Formula equivalent to v' = q* v q (unit quat).

      v' = v(2w²-1) - 2w(r×v) + 2r(r·v)   where r = [x,y,z]
    """
    w = q_wxyz[0]
    r = q_wxyz[1:]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(r, v) * (2.0 * w)
    c = r * (2.0 * np.dot(r, v))
    return a - b + c


class RLControllerNode(Node):
    def __init__(self):
        super().__init__("kutta_rl_controller")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("model_path", "/home/cuckylinux/ros2_ws/policy.onnx")
        model_path = self.get_parameter("model_path").get_parameter_value().string_value

        # ── Load ONNX policy ─────────────────────────────────────────────────
        self._session = None
        self._load_onnx(model_path)

        # ── Internal state (reset to standing pose defaults) ─────────────────
        self._joint_pos = np.zeros(ACTION_DIM, dtype=np.float32)
        self._joint_vel = np.zeros(ACTION_DIM, dtype=np.float32)
        self._ang_vel = np.zeros(3, dtype=np.float32)
        # Identity orientation → gravity points down in body frame
        self._projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._cmd = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Cache the joint index mapping (filled on first /joint_states message)
        self._joint_index: list[int] | None = None

        # ── Subscriptions ───────────────────────────────────────────────────
        self.create_subscription(JointState, "/joint_states", self._joint_states_cb, 10)
        self.create_subscription(Imu, "/imu/data", self._imu_cb, 10)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 10)

        # ── Publisher ────────────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Float64MultiArray, "/kutta/joint_commands", 10)

        # ── 50 Hz control loop ───────────────────────────────────────────────
        self.create_timer(1.0 / CTRL_HZ, self._control_loop)

        self.get_logger().info(f"kutta_rl_controller ready at {CTRL_HZ} Hz")

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_onnx(self, path: str) -> None:
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.get_logger().info(f"Loaded ONNX policy: {path}")
        except Exception as exc:
            self.get_logger().fatal(f"Failed to load ONNX model from '{path}': {exc}")
            raise SystemExit(1) from exc

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _joint_states_cb(self, msg: JointState) -> None:
        # Build index map on first message
        if self._joint_index is None:
            try:
                self._joint_index = [list(msg.name).index(n) for n in JOINT_NAMES]
            except ValueError as e:
                self.get_logger().warn(f"Joint name not found: {e}")
                return

        idx = self._joint_index
        if len(msg.position) >= 12:
            self._joint_pos[:] = [msg.position[i] for i in idx]
        if len(msg.velocity) >= 12:
            self._joint_vel[:] = [msg.velocity[i] for i in idx]

    def _imu_cb(self, msg: Imu) -> None:
        self._ang_vel[0] = msg.angular_velocity.x
        self._ang_vel[1] = msg.angular_velocity.y
        self._ang_vel[2] = msg.angular_velocity.z

        # Orientation quaternion [w, x, y, z] (ROS uses x,y,z,w in the message)
        q = np.array([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        ], dtype=np.float64)

        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q /= norm
            self._projected_gravity = _quat_rotate_inverse(
                q, np.array([0.0, 0.0, -1.0])
            ).astype(np.float32)

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._cmd[0] = msg.linear.x
        self._cmd[1] = msg.linear.y
        self._cmd[2] = msg.angular.z

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        if self._session is None:
            return

        # Build 45-element observation (same order as velocity_env_cfg.py actor_terms)
        obs = np.concatenate([
            self._ang_vel,           # [0:3]   base_ang_vel
            self._projected_gravity, # [3:6]   projected_gravity
            self._cmd,               # [6:9]   command [vx, vy, wz]
            self._joint_pos,         # [9:21]  joint_pos_rel (default=0)
            self._joint_vel,         # [21:33] joint_vel
            self._last_action,       # [33:45] last raw network output
        ], dtype=np.float32)[np.newaxis, :]  # shape (1, 45)

        # ONNX inference (normalisation is baked into the ONNX graph)
        raw_action: np.ndarray = self._session.run(None, {"obs": obs})[0][0]  # (12,)

        # Apply action scale and clip to joint limits
        # Matches: JointPositionActionCfg(scale=0.25, use_default_offset=True)
        #   target = default_pos + raw_action * scale = 0 + raw_action * 0.25
        target_pos = np.clip(raw_action * ACTION_SCALE, -JOINT_LIMIT, JOINT_LIMIT)

        # Store raw action for next obs (last_action = raw network output)
        self._last_action[:] = raw_action

        # Publish
        out = Float64MultiArray()
        out.data = target_pos.tolist()
        self._cmd_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RLControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
