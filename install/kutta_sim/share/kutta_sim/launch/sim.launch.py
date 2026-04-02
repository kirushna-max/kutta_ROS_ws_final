from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    xml_path_arg = DeclareLaunchArgument(
        "xml_path",
        default_value="/home/cuckylinux/ros2_ws/kutta.xml",
        description="Path to kutta.xml MJCF file",
    )
    viewer_arg = DeclareLaunchArgument(
        "viewer",
        default_value="true",
        description="Launch the MuJoCo passive viewer",
    )

    sim_node = Node(
        package="kutta_sim",
        executable="kutta_sim",
        name="kutta_sim",
        parameters=[{
            "xml_path": LaunchConfiguration("xml_path"),
            "viewer": LaunchConfiguration("viewer"),
        }],
        output="screen",
    )

    return LaunchDescription([xml_path_arg, viewer_arg, sim_node])
