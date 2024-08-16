#include "yolo_detector/yolo_detector.hpp"


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo_detector::YoloDetector>());
  rclcpp::shutdown();
  return 0;
}
