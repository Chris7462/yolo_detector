from os.path import join

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = join(
        get_package_share_directory('yolo_detector'), 'params',
        'yolo_detector.yaml'
    )

    yolo_detector_node = Node(
        package='yolo_detector',
        executable='yolo_detector_node',
        name='yolo_detector_node',
        output='screen',
        parameters=[params]
    )

    return LaunchDescription([
        yolo_detector_node
    ])
