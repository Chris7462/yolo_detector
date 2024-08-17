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

  conf_threshold_ = declare_parameter("conf_threshold", 0.5F);
  nms_threshold_ = declare_parameter("nms_threshold", 0.5F);
  nc_ = declare_parameter("nc", 80);
  double mean = declare_parameter("mean", 0.0);
  double scale = declare_parameter("scale", 1.0);
  int input_width = declare_parameter("width", 640);
  int input_height = declare_parameter("height", 640);
  bool swap_rb = declare_parameter("rgb", true);
  double padding_value = declare_parameter("padding_value", 114.0);
  cv::dnn::ImagePaddingMode padding_mode =
    static_cast<cv::dnn::ImagePaddingMode>(declare_parameter("padding_mode", 2));

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

  // load model
  load_net(model_file);

  // image pre-processing
  cv::Size size(input_width, input_height);
  img_params_.scalefactor = scale;
  img_params_.size = size;
  img_params_.mean = mean;
  img_params_.swapRB = swap_rb;
  img_params_.ddepth = CV_32F;
  img_params_.datalayout = cv::dnn::DNN_LAYOUT_NCHW;
  img_params_.paddingmode = padding_mode;
  img_params_.borderValue = padding_value;

  // rescale boxes back to original image
  cv::dnn::Image2BlobParams param_net;
  param_net_.scalefactor = scale;
  param_net_.size = size;
  param_net_.mean = mean;
  param_net_.swapRB = swap_rb;
  param_net_.paddingmode = padding_mode;

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

      try {
        cv::Mat cv_image = cv_bridge::toCvCopy(input_msg, "bgr8")->image;
        cv::Mat input_image = cv::dnn::blobFromImageWithParams(cv_image, img_params_);

        net_.setInput(cv_image);

        // forward
        std::vector<cv::Mat> outs;
        net_.forward(outs, net_.getUnconnectedOutLayersNames());

        std::vector<int> keep_class_ids;
        std::vector<float> keep_confidences;
        std::vector<cv::Rect2d> keep_boxes;

        // post processing
//      post_processing(outs, keep_class_ids, keep_confidences, keep_boxes,
//          conuThreshold, nmsThreshold,
//          yolo_model,
//          nc);


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

      } catch (cv_bridge::Exception & e) {
        RCLCPP_ERROR(get_logger(), "CV_Bridge exception: %s", e.what());
      }
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

void YoloDetector::post_processing(std::vector<cv::Mat>& outs,
  std::vector<int>& keep_classIds, std::vector<float>& keep_confidences,
  std::vector<cv::Rect2d>& keep_boxes, float conf_threshold,
  float iou_threshold, const std::string& model_name, const int nc)
{
  // Retrieve
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect2d> boxes;

  if (model_name == "yolov8" || model_name == "yolov10" || model_name == "yolov9") {
    cv::transposeND(outs[0], {0, 2, 1}, outs[0]);
  }

  if (model_name == "yolonas") {
    // outs contains 2 elemets of shape [1, 8400, 80] and [1, 8400, 4]. Concat them to get [1, 8400, 84]
    cv::Mat concat_out;
    // squeeze the first dimension
    outs[0] = outs[0].reshape(1, outs[0].size[1]);
    outs[1] = outs[1].reshape(1, outs[1].size[1]);
    cv::hconcat(outs[1], outs[0], concat_out);
    outs[0] = concat_out;
    // remove the second element
    outs.pop_back();
    // unsqueeze the first dimension
    outs[0] = outs[0].reshape(0, std::vector<int>{1, 8400, nc + 4});
  }

  // assert if last dim is 85 or 84
  CV_CheckEQ(outs[0].dims, 3, "Invalid output shape. The shape should be [1, #anchors, 85 or 84]");
  CV_CheckEQ((outs[0].size[2] == nc + 5 || outs[0].size[2] == 80 + 4), true, "Invalid output shape: ");

  for (auto preds : outs) {
    preds = preds.reshape(1, preds.size[1]); // [1, 8400, 85] -> [8400, 85]
    for (int i = 0; i < preds.rows; ++i) {
      // filter out non object
      float obj_conf = (model_name == "yolov8" || model_name == "yolonas" ||
        model_name == "yolov9" || model_name == "yolov10") ? 1.0f : preds.at<float>(i, 4);
      if (obj_conf < conf_threshold) {
        continue;
      }

      cv::Mat scores = preds.row(i).colRange(
        (model_name == "yolov8" || model_name == "yolonas" || model_name == "yolov9" || model_name == "yolov10") ? 4 : 5, preds.cols);
      double conf;
      cv::Point maxLoc;
      cv::minMaxLoc(scores, 0, &conf, 0, &maxLoc);

      conf = (model_name == "yolov8" || model_name == "yolonas" || model_name == "yolov9" || model_name == "yolov10") ? conf : conf * obj_conf;
      if (conf < conf_threshold) {
        continue;
      }

      // get bbox coords
      float* det = preds.ptr<float>(i);
      double cx = det[0];
      double cy = det[1];
      double w = det[2];
      double h = det[3];

      // [x1, y1, x2, y2]
      if (model_name == "yolonas" || model_name == "yolov10"){
        boxes.push_back(cv::Rect2d(cx, cy, w, h));
      } else {
        boxes.push_back(
          cv::Rect2d(cx - 0.5 * w, cy - 0.5 * h,cx + 0.5 * w, cy + 0.5 * h));
      }
      classIds.push_back(maxLoc.x);
      confidences.push_back(static_cast<float>(conf));
    }
  }

  // NMS
  std::vector<int> keep_idx;
  cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, keep_idx);

  for (auto i : keep_idx) {
    keep_classIds.push_back(classIds[i]);
    keep_confidences.push_back(confidences[i]);
    keep_boxes.push_back(boxes[i]);
  }
}


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
