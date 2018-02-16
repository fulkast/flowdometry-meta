#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32.h>

#include <device_2d.h>
#include <d_optical_and_ar_flow.h>
#include <utility_kernels.h>

#include <flowdometry_kernels.h>
#include <utils.h>

#include <opencv2/opencv.hpp>

#undef Success
#include <Eigen/Dense>

#include <iostream>

image_transport::Publisher debug_img_pub_;
int frame_count_;
cv::Mat texture;
util::Device2D<float>::Ptr d_float_frame_, d_depth_frame_;
util::Device2D<float3>::Ptr d_flow_depth_frame_;
cv::Mat depth_mask;

Eigen::Matrix<float, 3, 3, Eigen::RowMajor>  current_orientation;
Eigen::Vector3f current_position;
util::Device1D<float>::Ptr d_flow_ar_x_tmp_;
util::Device1D<uchar4>::Ptr d_flow_x_rgba_, d_flow_y_rgba_;
std::unique_ptr<vision::D_OpticalAndARFlow> d_optical_flow_;
std::vector<double> _A, _B, _dTdR;
ros::Publisher longitudinal_velocity_pub, path_pub;
nav_msgs::Path path_msg;
float lowpass_a;
float lowpass_b;
float lowpass_last_value;
float lowpass_value;
bool lowpass_ready;
float downscale_factor;
int image_height_, image_width_;
float fx, fy, cx, cy;
float tau,ts;

void composeNormalEquations(const float *CO) {
  _A = std::vector<double> (36,0);
  _B = std::vector<double> (6,0);
  _dTdR = std::vector<double> (6,0);

  _A[0] = CO[0];
  _A[1] = 0.0;
  _A[2] = CO[1];
  _A[3] = CO[2];
  _A[4] = CO[3];
  _A[5] = CO[4];
  _A[6] = 0.0;
  _A[7] = CO[0];
  _A[8] = CO[5];
  _A[9] = CO[6];
  _A[10] = -CO[2];
  _A[11] = CO[7];
  _A[12] = CO[1];
  _A[13] = CO[5];
  _A[14] = CO[8];
  _A[15] = CO[9];
  _A[16] = CO[10];
  _A[17] = 0.0;
  _A[18] = CO[2];
  _A[19] = CO[6];
  _A[20] = CO[9];
  _A[21] = CO[11];
  _A[22] = CO[12];
  _A[23] = CO[13];
  _A[24] = CO[3];
  _A[25] = -CO[2];
  _A[26] = CO[10];
  _A[27] = CO[12];
  _A[28] = CO[14];
  _A[29] = CO[15];
  _A[30] = CO[4];
  _A[31] = CO[7];
  _A[32] = 0.0;
  _A[33] = CO[13];
  _A[34] = CO[15];
  _A[35] = CO[16];

  for (int i = 0; i < 6; i++)
    _B[i] = CO[17 + i];
}

void solveNormalEquations(float *dTdR) {

  Eigen::Map<Eigen::Matrix<double, 6, 6> > A(_A.data());
  Eigen::Map<Eigen::Matrix<double, 6, 1> > B(_B.data());
  Eigen::Map<Eigen::Matrix<double, 6, 1> > double_dTdR(_dTdR.data());

  double_dTdR = A.ldlt().solve(B);

  Eigen::Map<Eigen::Matrix<float, 6, 1> > float_dTdR(dTdR);
  float_dTdR = double_dTdR.cast<float>();
}

void initVariables() {
  // load ROS parameters
  downscale_factor = 1.;
  ros::param::get("flowdometry/downscale_factor", downscale_factor);

  // downscale image dimensions
  image_height_ *= downscale_factor;
  image_width_ *= downscale_factor;

  if (!ros::param::get("flowdometry/camera_info/fx", fx) ||
      !ros::param::get("flowdometry/camera_info/fy", fy) ||
      !ros::param::get("flowdometry/camera_info/cx", cx) ||
      !ros::param::get("flowdometry/camera_info/cy", cy)
    ) {
      throw std::runtime_error(
          std::string("flowdometry camera intrinsics (camera_info) \
                        are require on ROS parameter server\n"));
  }

  tau = 3.;
  ts = 1.;

  ros::param::get("flowdometry/longitudinal_velocity_filter/lowpass_tau", tau);
  ros::param::get("flowdometry/longitudinal_velocity_filter/lowpass_ts", ts);


  // Initialize position and orientation
  current_orientation = Eigen::Matrix<float, 3, 3>::Identity();
  current_position << 0., 0., 0. ;

  lowpass_a = 1. / (tau / ts + 1.);
  lowpass_b = tau / ts / (tau / ts + 1.);

  d_float_frame_ =
    util::Device2D<float>::Ptr(new util::Device2D<float>(image_width_,
                                              image_height_));
  d_depth_frame_ =
      util::Device2D<float>::Ptr(new util::Device2D<float>(image_width_, image_height_));

  d_flow_depth_frame_ =
      util::Device2D<float3>::Ptr(new util::Device2D<float3>(image_width_, image_height_));

  d_flow_ar_x_tmp_ =
      util::Device1D<float>::Ptr(new util::Device1D<float>(image_width_ * image_height_));
  d_flow_x_rgba_ =
      util::Device1D<uchar4>::Ptr(new util::Device1D<uchar4>(image_width_ * image_height_));
  d_flow_y_rgba_ =
      util::Device1D<uchar4>::Ptr(new util::Device1D<uchar4>(image_width_ * image_height_));


  vision::D_OpticalAndARFlow::Parameters parameters_flow;
  parameters_flow.n_scales_ = 4;
  d_optical_flow_ = std::unique_ptr<vision::D_OpticalAndARFlow>{
    new vision::D_OpticalAndARFlow(*d_float_frame_, parameters_flow)
  };

  lowpass_last_value = 0.;
  lowpass_ready = false;
}


void depthAndColorCb(
          const sensor_msgs::ImageConstPtr &rgb_msg,
          const sensor_msgs::ImageConstPtr &depth_msg)
{

  cv_bridge::CvImageConstPtr cv_rgb_ptr, cv_depth_ptr;

  try {
    if (rgb_msg->encoding == "8UC1")
      cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg);
    else
      cv_rgb_ptr = cv_bridge::toCvShare(rgb_msg, "mono8");
    }   catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

  cv_depth_ptr = cv_bridge::toCvShare(depth_msg);
  cv::Mat depth_image = cv_depth_ptr->image.clone();

  if (d_float_frame_ == nullptr) {
    image_height_ = std::min(rgb_msg->height, depth_msg->height);
    image_width_ = std::min(rgb_msg->width, depth_msg->width);
    initVariables();
  }

  // computer optical flow
  // convert image to float
  cv::Mat image_float;
  cv_rgb_ptr->image.convertTo(image_float, CV_32FC1);
  cv::resize(image_float, image_float, cv::Size(image_width_, image_height_));

  // copy new image to device
  checkCudaErrors(cudaMemcpy2D(d_float_frame_->data(), d_float_frame_->pitch(),
               image_float.data, image_width_ * sizeof(float),
               image_width_ * sizeof(float), image_height_,
               cudaMemcpyHostToDevice));

 // update optical flow
 d_optical_flow_->addImageReal(*d_float_frame_);
 d_optical_flow_->updateOpticalFlowReal();

 // mark valid optical flow and depth buffer
 // send depth to device
 cv::Mat depth_image_float = cv::Mat::zeros(image_height_, image_width_, CV_32FC1);
 depth_image.copyTo(depth_image_float, depth_mask);
 depth_image_float.convertTo(depth_image_float, CV_32FC1);

 // std::cout << "depth image float " << depth_image_float << std::endl;

 cv::resize(depth_image_float, depth_image_float, cv::Size(image_width_, image_height_));

 checkCudaErrors(cudaMemcpy2D(d_depth_frame_->data(), d_depth_frame_->pitch(),
              depth_image_float.data, image_width_ * sizeof(float),
              image_width_ * sizeof(float), image_height_,
              cudaMemcpyHostToDevice));

 // fill flow depth image

 int intermediate_size = image_height_;
 int n_threads = image_width_;
 int n_blocks = image_height_;

 float * d_intermediate_reduction_result;

 checkCudaErrors( cudaMalloc(&d_intermediate_reduction_result,
                         sizeof(float) * intermediate_size * 23) );

 std::vector<float> A(23);

 normal_eqs_flow_GPU(&A[0], d_intermediate_reduction_result,
   d_optical_flow_->getOpticalFlowX().data(),
   d_optical_flow_->getOpticalFlowY().data(),
   d_depth_frame_->data(), fx * downscale_factor, fy* downscale_factor,
                            cx* downscale_factor, cy* downscale_factor,image_height_,
                                     image_width_);

 checkCudaErrors(cudaFree(d_intermediate_reduction_result));

 composeNormalEquations(&A[0]);
 std::vector<float> dTdR(6,0);
 solveNormalEquations(&dTdR[0]);


 std_msgs::Float32 msg;
 if (lowpass_ready) {
  lowpass_value = -lowpass_a * dTdR[2] + lowpass_b * lowpass_last_value;
  lowpass_last_value = lowpass_value;

  Eigen::Map<Eigen::Matrix<float, 3, 1> > delta_T(dTdR.data());
  Eigen::Map<const Eigen::Vector3f> rot_axis_angle(&dTdR[3]);

  // update current position and orientation
  current_position = current_orientation * -delta_T + current_position;

  std::cout << "delta t " << -delta_T(2) << std::endl;

  float angle = -rot_axis_angle.norm();
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot_mat;

  if (angle < 1e-15) {
    // identity matrix
    rot_mat = Eigen::Matrix<float, 3, 3>::Identity();
  } else {
    rot_mat = Eigen::AngleAxis<float>(angle, rot_axis_angle / angle).toRotationMatrix();
  }

  current_orientation = current_orientation * rot_mat;
  // std::cout << "current position: " << current_position << std::endl;

  Eigen::Quaternionf current_quaternion(current_orientation);

  geometry_msgs::PoseStamped pose_stamped_msg;
  path_msg.header = rgb_msg->header;
  path_msg.header.frame_id = "/map";
  pose_stamped_msg.header = rgb_msg->header;
  pose_stamped_msg.header.frame_id = "/map";
  pose_stamped_msg.pose.position.x = current_position.x();
  pose_stamped_msg.pose.position.y = current_position.y();
  pose_stamped_msg.pose.position.z = current_position.z();
  pose_stamped_msg.pose.orientation.x = current_quaternion.x();
  pose_stamped_msg.pose.orientation.y = current_quaternion.y();
  pose_stamped_msg.pose.orientation.z = current_quaternion.z();
  pose_stamped_msg.pose.orientation.w = current_quaternion.w();
  path_msg.poses.resize(frame_count_+1);
  path_msg.poses[frame_count_] = pose_stamped_msg;
  path_pub.publish(path_msg);

 } else {
   lowpass_last_value = -dTdR[2];
   lowpass_ready = true;
 }
 msg.data = lowpass_value;
 longitudinal_velocity_pub.publish(msg);

  frame_count_++;
}


int main(int argc, char  *argv[]) {
  ros::init(argc, argv, "flowdometry_node");
  frame_count_ = 0;
  // Subscriptions
  boost::shared_ptr<image_transport::ImageTransport> rgb_it_, depth_it_;
  image_transport::SubscriberFilter sub_depth_, sub_rgb_;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image> SyncPolicyRGBD;
  typedef message_filters::Synchronizer<SyncPolicyRGBD> SynchronizerRGBD;
  boost::shared_ptr<SynchronizerRGBD> sync_rgbd_;

  boost::shared_ptr<image_transport::ImageTransport> debug_img_it_;

  bool compressed_streams = false;

  ros::NodeHandle nh_;

  longitudinal_velocity_pub = nh_.advertise<std_msgs::Float32>("longitudinal_velocity", 100);
  path_pub = nh_.advertise<nav_msgs::Path>("trajectory", 10);

  image_transport::TransportHints rgb_hint, depth_hint;
  if (compressed_streams) {
    rgb_hint = image_transport::TransportHints("compressed");
    depth_hint = image_transport::TransportHints("compressedDepth");
  } else {
    rgb_hint = image_transport::TransportHints("raw");
    depth_hint = image_transport::TransportHints("raw");
  }

  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_rgb_.subscribe(*rgb_it_, "rgb", 2, rgb_hint);
  depth_it_.reset(new image_transport::ImageTransport(nh_));
  sub_depth_.subscribe(*depth_it_, "depth", 2, depth_hint);
  sync_rgbd_.reset(new SynchronizerRGBD(SyncPolicyRGBD(5),
                                        sub_rgb_,
                                        sub_depth_));
  sync_rgbd_->registerCallback(
      boost::bind(&depthAndColorCb, _1, _2));

  debug_img_it_.reset(new image_transport::ImageTransport(nh_));
  debug_img_pub_ = debug_img_it_->advertise("/flowdometry/image", 1);

  ros::spin();

  return 0;
}
