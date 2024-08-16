#pragma once

// C++ header
#include <queue>
#include <mutex>
//#include <memory>
#include <vector>
#include <string>
#include <filesystem>

// openCV header
//#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

// ROS header
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>


namespace yolo_detector
{

namespace fs = std::filesystem;

//  struct Detection
//  {
//    int class_id;
//    float confidence;
//    cv::Rect box;
//  };

class YoloDetector : public rclcpp::Node
{
public:
  YoloDetector();
  ~YoloDetector() = default;

private:
  void img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

  void timer_callback();
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr yolo_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::queue<sensor_msgs::msg::Image::SharedPtr> img_buff_;

  std::mutex mtx_;

  bool get_classes(fs::path class_file);
  std::vector<std::string> classes_;

  void load_net(fs::path model_file);

//  cv::Mat format_yolov5(const cv::Mat & source);
//  void detect(cv::Mat & image, std::vector<Detection> & output);

  cv::dnn::Net net_;
};

} // namespace yolo_detector
