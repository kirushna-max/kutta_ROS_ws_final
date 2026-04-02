"""kutta_sim — ROS 2 MuJoCo simulation node.

Loads kutta.xml, steps the physics at 200 Hz (same timestep as training),
publishes sensor data, and accepts joint position commands.

This mirrors the simulation side of the rl_mjlab training environment:
  - MuJoCo timestep = 0.005 s  (option timestep in kutta.xml)
  - Policy decimation = 4       → controller runs at 50 Hz
  - Position actuators (kp=20, forcerange=±10 Nm) match kutta.xml defaults

Subscribed topics
-----------------
/kutta/joint_commands  std_msgs/Float64MultiArray  12 target positions (rad)
  revolute1..12 in order: FL-hip, FL-thigh, FL-knee,
                           RL-hip, RL-thigh, RL-knee,
                           FR-hip, FR-thigh, FR-knee,
                           RR-hip, RR-thigh, RR-knee

Published topics
----------------
/joint_states  sensor_msgs/JointState   revolute1..12 pos + vel + torque
/imu/data      sensor_msgs/Imu          angular velocity, orientation, linear accel

Parameters
----------
xml_path  (string)  Path to kutta.xml  [default: /home/cuckylinux/ros2_ws/kutta.xml]
viewer    (bool)    Launch MuJoCo passive viewer  [default: true]
"""

import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

JOINT_NAMES = [f"revolute{i}" for i in range(1, 13)]

# MuJoCo freejoint layout in qpos/qvel:
#   qpos[0:3]  = base position (x,y,z)
#   qpos[3:7]  = base quaternion [w,x,y,z]
#   qpos[7:19] = revolute1..12 angles (rad)
#
#   qvel[0:3]  = base linear velocity
#   qvel[3:6]  = base angular velocity
#   qvel[6:18] = revolute1..12 velocities (rad/s)
_QPOS_JOINT_SLICE = slice(7, 19)
_QVEL_JOINT_SLICE = slice(6, 18)
_QPOS_QUAT_SLICE  = slice(3, 7)   # [w, x, y, z]


class KuttaSimNode(Node):
    def __init__(self):
        super().__init__("kutta_sim")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("xml_path", "/home/cuckylinux/ros2_ws/kutta.xml")
        self.declare_parameter("viewer", True)

        xml_path = self.get_parameter("xml_path").get_parameter_value().string_value
        use_viewer = self.get_parameter("viewer").get_parameter_value().bool_value

        # ── Load MuJoCo model ────────────────────────────────────────────────
        self.get_logger().info(f"Loading MJCF: {xml_path}")
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)

        # Forward pass to initialise sensor values
        mujoco.mj_forward(self._model, self._data)

        # ── Sensor IDs (looked up once for speed) ────────────────────────────
        self._gyro_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
        )
        self._acc_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_lin_acc"
        )
        # Tau (actuator force) sensor start address — optional, for effort field
        # We look up tau_revolute1 as the first of 12 contiguous torque sensors
        self._tau_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, "tau_revolute1"
        )

        # ── Command state ────────────────────────────────────────────────────
        self._target_pos = np.zeros(12, dtype=np.float64)
        self._lock = threading.Lock()

        # ── ROS interfaces ───────────────────────────────────────────────────
        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._imu_pub = self.create_publisher(Imu, "/imu/data", 10)

        self.create_subscription(
            Float64MultiArray,
            "/kutta/joint_commands",
            self._cmd_cb,
            10,
        )

        # ── Simulation loop in a dedicated thread ────────────────────────────
        # ROS timers at 5 ms can drift on some systems; using a tight real-time
        # loop with time.sleep is more accurate and avoids callback stacking.
        self._running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

        # ── Optional passive viewer ───────────────────────────────────────────
        self._viewer = None
        if use_viewer:
            self._launch_viewer()

        self.get_logger().info(
            f"kutta_sim ready — sim dt={self._model.opt.timestep*1e3:.1f} ms, "
            f"viewer={'on' if self._viewer else 'off'}"
        )

    # ── Viewer ───────────────────────────────────────────────────────────────

    def _launch_viewer(self) -> None:
        try:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self.get_logger().info("MuJoCo passive viewer launched")
        except Exception as exc:
            self.get_logger().warn(f"Could not launch viewer: {exc}")

    # ── Command callback ─────────────────────────────────────────────────────

    def _cmd_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) == 12:
            with self._lock:
                self._target_pos[:] = msg.data

    # ── Simulation loop (dedicated thread) ───────────────────────────────────

    def _sim_loop(self) -> None:
        """Real-time physics loop.

        Runs mj_step at exactly model.opt.timestep wall-clock seconds per step.
        Uses a sleep-then-busywait strategy: sleep most of the interval, then
        spin on monotonic clock for the last fraction to get sub-millisecond
        accuracy without burning CPU the whole time.
        """
        dt = float(self._model.opt.timestep)   # 0.005 s = 200 Hz
        BUSYWAIT_MARGIN = 0.0005               # busywait the last 0.5 ms

        next_step_time = time.monotonic()

        while self._running:
            next_step_time += dt

            # Coarse sleep — give up most of the remaining time
            sleep_time = next_step_time - time.monotonic() - BUSYWAIT_MARGIN
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Fine busywait for the last 0.5 ms
            while time.monotonic() < next_step_time:
                pass

            # Apply latest joint position targets
            with self._lock:
                self._data.ctrl[:] = self._target_pos

            # Advance physics
            mujoco.mj_step(self._model, self._data)

            # Sync viewer (non-blocking)
            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()

            if self._running:
                try:
                    self._publish_sensors()
                except Exception:
                    # Publisher context invalid during shutdown — exit cleanly
                    break

    # ── Sensor publishing ────────────────────────────────────────────────────

    def _publish_sensors(self) -> None:
        stamp = self.get_clock().now().to_msg()

        # ── JointState ───────────────────────────────────────────────────────
        js = JointState()
        js.header.stamp = stamp
        js.header.frame_id = "base_link"
        js.name = JOINT_NAMES
        js.position = self._data.qpos[_QPOS_JOINT_SLICE].tolist()
        js.velocity = self._data.qvel[_QVEL_JOINT_SLICE].tolist()

        # Actuator torques from MuJoCo sensor data (12 contiguous readings)
        if self._tau_id >= 0:
            tau_start = int(self._model.sensor_adr[self._tau_id])
            js.effort = self._data.sensordata[tau_start: tau_start + 12].tolist()

        self._joint_pub.publish(js)

        # ── IMU ──────────────────────────────────────────────────────────────
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "base_link"

        # Angular velocity from gyro sensor (body frame)
        if self._gyro_id >= 0:
            ang_adr = int(self._model.sensor_adr[self._gyro_id])
            ang = self._data.sensordata[ang_adr: ang_adr + 3]
        else:
            ang = self._data.qvel[3:6]  # fallback: body angular velocity

        imu.angular_velocity.x = float(ang[0])
        imu.angular_velocity.y = float(ang[1])
        imu.angular_velocity.z = float(ang[2])

        # Orientation: freejoint qpos[3:7] = [w, x, y, z]
        # Publish as ROS convention (x, y, z, w in message fields)
        q = self._data.qpos[_QPOS_QUAT_SLICE]  # [w, x, y, z]
        imu.orientation.w = float(q[0])
        imu.orientation.x = float(q[1])
        imu.orientation.y = float(q[2])
        imu.orientation.z = float(q[3])

        # Linear acceleration from accelerometer sensor
        if self._acc_id >= 0:
            acc_adr = int(self._model.sensor_adr[self._acc_id])
            acc = self._data.sensordata[acc_adr: acc_adr + 3]
        else:
            acc = np.zeros(3)

        imu.linear_acceleration.x = float(acc[0])
        imu.linear_acceleration.y = float(acc[1])
        imu.linear_acceleration.z = float(acc[2])

        # Unknown covariances (set to -1 = unknown per REP-145)
        imu.orientation_covariance[0] = -1.0
        imu.angular_velocity_covariance[0] = -1.0
        imu.linear_acceleration_covariance[0] = -1.0

        self._imu_pub.publish(imu)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._running = False
        self._sim_thread.join(timeout=2.0)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KuttaSimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
