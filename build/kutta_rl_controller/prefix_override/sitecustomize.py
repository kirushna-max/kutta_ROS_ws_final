import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cuckylinux/ros2_ws/install/kutta_rl_controller'
