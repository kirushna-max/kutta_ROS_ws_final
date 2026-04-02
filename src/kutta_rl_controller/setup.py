from setuptools import find_packages, setup

package_name = "kutta_rl_controller"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/controller.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="RL policy controller for the Kutta quadruped",
    license="MIT",
    entry_points={
        "console_scripts": [
            "rl_controller = kutta_rl_controller.rl_controller_node:main",
        ],
    },
)
