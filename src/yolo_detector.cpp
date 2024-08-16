// C++ header
#include <fstream>
#include <chrono>

#include <opencv2/core/cuda.hpp>

// ROS header
#include <cv_bridge/cv_bridge.h>

// local header
#include "yolo_detector/yolo_detector.hpp"


namespace yolo_detector
{

using namespace std::chrono_literals;

YoloDetector::YoloDetector()
: Node("yolo_detector_node")
{
  fs::path model_path = declare_parameter("model_path", fs::path());
  std::string yolo_model = declare_parameter("yolo_model", std::string());
  fs::path model_file = model_path / declare_parameter("model_file", std::string());
  fs::path classes_file = model_path / declare_parameter("classes_file", std::string());
  int nc = declare_parameter("nc", 80);
  double conf_threshold = declare_parameter("conf_threshold", 0.5);
  double nms_threshold = declare_parameter("nms_threshold", 0.5);

  //![preprocess_params]
  double mean = declare_parameter("mean", 0.0);
  double scale = declare_parameter("scale", 1.0);
  int input_width = declare_parameter("width", 640);
  int input_height = declare_parameter("height", 640);
  bool swap_rb = declare_parameter("rgb", true);
  double padding_value = declare_parameter("padding_value", 114);
  cv::dnn::ImagePaddingMode padding_mode = static_cast<cv::dnn::ImagePaddingMode>(declare_parameter("padding_mode", 2));


  //![preprocess_params]

//  rgb: True # Indicate that model works with RGB input images instead BGR ones.
//  padvalue: 114.0 # padding value.
//  paddingmode: 2 # Choose one of computation backends:
//                 # "0: resize to required input size without extra processing",
//                 # "1: Image will be cropped after resize",
//                 # "2: Resize image to the desired size while preserving the aspect ratio of original image"
//  backend: 0 # Choose one of computation backends:
//             # "0: automatically (by default)"",
//             # "1: Halide language (http://halide-lang.org/)",
//             # "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit)",
//             # "3: OpenCV implementation",
//             # "4: VKCOM",
//             # "5: CUDA"
//  target: 0 # Choose one of target computation devices:
//            # "0: CPU target (by default)",
//            # "1: OpenCL",
//            # "2: OpenCL fp16 (half-float precision)",
//            # "3: VPU",
//            # "4: Vulkan",
//            # "6: CUDA",
//            # "7: CUDA fp16 (half-float preprocess)"
//  async: 0  # Number of asynchronous forwards at the same time.
//            # "Choose 0 for synchronous mode

  // check if yolo model is valid
  if (yolo_model != "yolov5" && yolo_model != "yolov6" &&
      yolo_model != "yolov7" && yolo_model != "yolov8" &&
      yolo_model != "yolov9" && yolo_model != "yolov10") {
    RCLCPP_ERROR(get_logger(), "Invalid yolo model: %s", yolo_model.c_str());
    rclcpp::shutdown();
  }

  if (!fs::exists(model_file)) {
    RCLCPP_ERROR(get_logger(), "Load model failed");
    rclcpp::shutdown();
  }

  if (!get_classes(classes_file)) {
    RCLCPP_ERROR(get_logger(), "Load classes list failed");
    rclcpp::shutdown();
  }

  load_net(model_file);

//// pre-process call
//cv::Size size()

  rclcpp::QoS qos(10);
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "kitti/camera/color/left/image_raw", qos, std::bind(
      &YoloDetector::img_callback, this, std::placeholders::_1));

  yolo_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
    "yolo_detector", qos);

  timer_ = this->create_wall_timer(
    25ms, std::bind(&YoloDetector::timer_callback, this));
}

void YoloDetector::img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);
  img_buff_.push(msg);
}

void YoloDetector::timer_callback()
{
  if (!img_buff_.empty()) {
    rclcpp::Time current_time = rclcpp::Node::now();
    mtx_.lock();
    if ((current_time - rclcpp::Time(img_buff_.front()->header.stamp)).seconds() > 0.1) {
      // time sync has problem
      RCLCPP_WARN(get_logger(), "Timestamp unaligned, please check your IMAGE data.");
      img_buff_.pop();
      mtx_.unlock();
    } else {
      auto input_msg = img_buff_.front();
      img_buff_.pop();
      mtx_.unlock();

//    try {
//      cv::Mat cv_image = cv_bridge::toCvCopy(input_msg, "bgr8")->image;
//      std::vector<Detection> detections;
//      detect(cv_image, detections);

//      for (const auto & detection : detections) {
//        auto box = detection.box;
//        auto class_id = detection.class_id;
//        auto color = colors[class_id % colors.size()];

//        cv::rectangle(cv_image, box, color, 2);
//        cv::rectangle(
//          cv_image, cv::Point(box.x, box.y - 10.0),
//          cv::Point(box.x + box.width, box.y), color, cv::FILLED);
//        cv::putText(
//          cv_image, class_list_[class_id].c_str(), cv::Point(box.x, box.y - 5.0),
//          cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(0.0, 0.0, 0.0));
//      }

//      // Convert OpenCV image to ROS Image message
//      auto out_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_image).toImageMsg();
//      out_msg->header.frame_id = "cam2_link";
//      out_msg->header.stamp = current_time;
//      yolo_pub_->publish(*out_msg);

//    } catch (cv_bridge::Exception & e) {
//      RCLCPP_ERROR(get_logger(), "CV_Bridge exception: %s", e.what());
//    }
    }
  }
}

bool YoloDetector::get_classes(fs::path classes_file)
{
  std::ifstream ifs(classes_file.c_str());
  if (!ifs.is_open()) {
    RCLCPP_ERROR(get_logger(), "File %s not found", classes_file.c_str());
    return false;
  } else {
    std::string line;
    while (std::getline(ifs, line)) {
      classes_.push_back(line);
    }
    return true;
  }
}

void YoloDetector::load_net(fs::path model_file)
{
  net_ = cv::dnn::readNet(model_file);
  if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
    RCLCPP_INFO(get_logger(), "CUDA is available. Attempy to use CUDA");
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
  } else {
    RCLCPP_INFO(get_logger(), "No CUDA-enabled devices found. Running on CPU");
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
}

//  cv::Mat YoloDetector::format_yolov5(const cv::Mat & source)
//  {
//    int col = source.cols;
//    int row = source.rows;
//    int max = std::max(col, row);
//    cv::Mat result = cv::Mat::zeros(max, max, CV_8UC3);
//    source.copyTo(result(cv::Rect(0, 0, col, row)));

//    return result;
//  }

//  void YoloDetector::detect(cv::Mat & image, std::vector<Detection> & output)
//  {
//    auto input_image = format_yolov5(image);

//    cv::Mat blob;
//    cv::dnn::blobFromImage(
//      input_image, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
//    net_.setInput(blob);

//    std::vector<cv::Mat> outputs;
//    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

//    float x_factor = input_image.cols / INPUT_WIDTH;
//    float y_factor = input_image.rows / INPUT_HEIGHT;

//    float * data = (float *)outputs[0].data;

//    std::vector<int> class_ids;
//    std::vector<float> confidences;
//    std::vector<cv::Rect> boxes;

//    for (int i = 0; i < OUTPUT_ROWS; ++i) {
//      float confidence = data[4];
//      if (confidence >= CONFIDENCE_THRESHOLD) {
//        float * classes_scores = data + 5;
//        cv::Mat scores(1, class_list_.size(), CV_32FC1, classes_scores);
//        cv::Point class_id;
//        double max_class_score;
//        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
//        if (max_class_score > SCORE_THRESHOLD) {
//          confidences.push_back(confidence);
//          class_ids.push_back(class_id.x);

//          float x = data[0];
//          float y = data[1];
//          float w = data[2];
//          float h = data[3];
//          int left = int((x - 0.5 * w) * x_factor);
//          int top = int((y - 0.5 * h) * y_factor);
//          int width = int(w * x_factor);
//          int height = int(h * y_factor);
//          boxes.push_back(cv::Rect(left, top, width, height));
//        }
//      }
//      data += CLASS_DIMENSIONS;
//    }

//    std::vector<int> nms_result;
//    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
//    for (size_t i = 0; i < nms_result.size(); ++i) {
//      int idx = nms_result[i];
//      Detection result;
//      result.class_id = class_ids[idx];
//      result.confidence = confidences[idx];
//      result.box = boxes[idx];
//      output.push_back(result);
//    }
//  }

} // namespace yolo_detector
