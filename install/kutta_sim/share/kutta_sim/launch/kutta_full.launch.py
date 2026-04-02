"""Launch both kutta_sim and kutta_rl_controller together.

Usage:
  ros2 launch kutta_sim kutta_full.launch.py
  ros2 launch kutta_sim kutta_full.launch.py viewer:=false
  ros2 launch kutta_sim kutta_full.launch.py model_path:=/path/to/policy.onnx
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "xml_path",
            default_value="/home/cuckylinux/ros2_ws/kutta.xml",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value="/home/cuckylinux/ros2_ws/policy.onnx",
        ),
        DeclareLaunchArgument(
            "viewer",
            default_value="true",
            description="Show the MuJoCo passive viewer",
        ),
        Node(
            package="kutta_sim",
            executable="kutta_sim",
            name="kutta_sim",
            parameters=[{
                "xml_path": LaunchConfiguration("xml_path"),
                "viewer": LaunchConfiguration("viewer"),
            }],
            output="screen",
        ),
        Node(
            package="kutta_rl_controller",
            executable="rl_controller",
            name="kutta_rl_controller",
            parameters=[{
                "model_path": LaunchConfiguration("model_path"),
            }],
            output="screen",
        ),
    ])
