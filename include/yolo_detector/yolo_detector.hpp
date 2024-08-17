#pragma once

// C++ header
#include <queue>
#include <mutex>
#include <vector>
#include <string>
#include <filesystem>

// openCV header
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

// ROS header
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>


namespace yolo_detector
{

namespace fs = std::filesystem;

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

  float conf_threshold_;
  float nms_threshold_;
  int nc_;

  cv::dnn::Image2BlobParams img_params_;
  cv::dnn::Image2BlobParams param_net_;

  std::vector<cv::Rect> boxes_;

  cv::dnn::Net net_;

  void post_processing(std::vector<cv::Mat>& outs,
    std::vector<int>& keep_classIds, std::vector<float>& keep_confidences,
    std::vector<cv::Rect2d>& keep_boxes, float conf_threshold,
    float iou_threshold, const std::string& model_name, const int nc = 80);
};

} // namespace yolo_detector
