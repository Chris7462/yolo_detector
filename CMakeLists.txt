cmake_minimum_required(VERSION 3.8)
project(yolo_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3 -Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# message(STATUS "OpenCV include dir =====> ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OpenCV lib dir =====> ${OpenCV_INSTALL_PATH}")
# message(STATUS "OpenCV version =====> ${OpenCV_VERSION}")
# https://github.com/opencv/opencv/blob/master/cmake/templates/OpenCVConfig.cmake.in#L231

add_executable(yolo_detector_node
  src/yolo_detector_node.cpp
  src/yolo_detector.cpp)

target_include_directories(yolo_detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS})

ament_target_dependencies(yolo_detector_node PUBLIC
  rclcpp
  sensor_msgs
  cv_bridge)

target_link_libraries(yolo_detector_node PUBLIC
  ${OpenCV_LIBS})
# opencv_imgproc
# opencv_imgcodecs
# opencv_dnn)

target_compile_features(yolo_detector_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS yolo_detector_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY params
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
