from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="/home/cuckylinux/ros2_ws/policy.onnx",
        description="Path to the ONNX policy file",
    )

    controller_node = Node(
        package="kutta_rl_controller",
        executable="rl_controller",
        name="kutta_rl_controller",
        parameters=[{"model_path": LaunchConfiguration("model_path")}],
        output="screen",
    )

    return LaunchDescription([model_path_arg, controller_node])
