#include "openpose_wrapper.h"

#include <image_recognition_msgs/Recognize.h>
#include <ros/node_handle.h>
#include <openpose/headers.hpp>
#include <opencv2/opencv.hpp>

std::shared_ptr<OpenposeWrapper> g_openpose_wrapper;
std::string g_save_images_folder = "";
bool g_publish_to_topic = true;
ros::Publisher g_pub;

//!
//! \brief getParam Get parameter from node handle
//! \param nh The nodehandle
//! \param param_name Key string
//! \param default_value Default value if not found
//! \return The parameter value
//!
template <typename T>
T getParam(const ros::NodeHandle& nh, const std::string& param_name, T default_value)
{
  T value;
  if (nh.hasParam(param_name))
  {
    nh.getParam(param_name, value);
  }
  else
  {
    ROS_WARN_STREAM("Parameter '" << param_name << "' not found, defaults to '" << default_value << "'");
    value = default_value;
  }
  return value;
}

//!
//! \brief getTimeAsString Returns the current time as a string using the python strftime syntax
//! \param format_string The python strftime syntax
//! \return Returns the date string
//!
std::string getTimeAsString(std::string format_string)
{
  time_t t = time(0);   // get time now
  struct tm* timeinfo = localtime( & t );

  format_string += '\a'; //force at least one character in the result
  std::string buffer;
  buffer.resize(format_string.size());
  int len = strftime(&buffer[0], buffer.size(), format_string.c_str(), timeinfo);
  while (len == 0) {
    buffer.resize(buffer.size()*2);
    len = strftime(&buffer[0], buffer.size(), format_string.c_str(), timeinfo);
  }
  buffer.resize(len-1); //remove that trailing '\a'
  return buffer;
}


bool visualizePose(const cv::Mat& image, op::Array<float> poseKeypoints)
{
  cv::Mat overlayed_image;

  ros::Time start = ros::Time::now();
  if (!g_openpose_wrapper->visualizePose(image, poseKeypoints, overlayed_image))
  {
    ROS_ERROR("g_openpose_wrapper_->visualizePose failed!");
    return false;
  }
  ROS_INFO("g_openpose_wrapper->visualizePose took %.3f seconds", (ros::Time::now() - start).toSec());

  // Write to disk
  if (!g_save_images_folder.empty())
  {
    std::string output_filepath = g_save_images_folder + "/" + getTimeAsString("%Y-%m-%d-%H-%M-%S") + "_openpose_ros.jpg";
    ROS_INFO("Writing output to %s", output_filepath.c_str());
    cv::imwrite(output_filepath, overlayed_image);
  }

  // Publish to topic
  if (g_publish_to_topic)
  {
    try
    {
      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", overlayed_image).toImageMsg();
      g_pub.publish(msg);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("Failed to published overlayed image; cv_bridge exception: %s", e.what());
    }
  }

  return true;
}

//!
//! \brief detectPoses Detects poses in an input image and optionally writes the detection image to topic and or disk
//! \param image Source image
//! \param recognitions Recognition reference
//! \return True if success, False otherwise
//!
bool detectPoses(const cv::Mat& image, std::vector<image_recognition_msgs::Recognition>& recognitions)
{
  cv::Mat overlayed_image;

  ros::Time start = ros::Time::now();
  if (!g_openpose_wrapper->detectPoses(image, recognitions, overlayed_image))
  {
    ROS_ERROR("g_openpose_wrapper_->detectPoses failed!");
    return false;
  }
  ROS_INFO("g_openpose_wrapper->detectPoses took %.3f seconds", (ros::Time::now() - start).toSec());

  // Write to disk
  if (!g_save_images_folder.empty())
  {
    std::string output_filepath = g_save_images_folder + "/" + getTimeAsString("%Y-%m-%d-%H-%M-%S") + "_openpose_ros.jpg";
    ROS_INFO("Writing output to %s", output_filepath.c_str());
    cv::imwrite(output_filepath, overlayed_image);
  }

  // Publish to topic
  if (g_publish_to_topic)
  {
    try
    {
      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", overlayed_image).toImageMsg();
      g_pub.publish(msg);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("Failed to published overlayed image; cv_bridge exception: %s", e.what());
    }
  }

  return true;
}

bool visualizePoseCallback(image_recognition_msgs::Recognize::Request& req, image_recognition_msgs::Recognize::Response& res)
{
  // ROS_INFO("visualizePoseCallback");
  op::Array<float> poseKeypoints;

  // Convert ROS message to opencv image
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(req.image, req.image.encoding);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("visualizePoseCallback cv_bridge exception: %s", e.what());
    return false;
  }
  cv::Mat image = cv_ptr->image;
  if(image.empty())
  {
    ROS_ERROR("Empty image!");
    return false;
  }

  double width_factor = (double) image.cols / image.cols;
  double height_factor = (double) image.rows / image.rows;
  double scale_factor = std::fmax(width_factor, height_factor);
  for (int index=0; index < res.recognitions.size(); index++)
  {
    poseKeypoints[3 * index] = res.recognitions[index].roi.x_offset / scale_factor;
    poseKeypoints[3 * index + 1] = res.recognitions[index].roi.y_offset / scale_factor;
    poseKeypoints[3 * index + 2] = res.recognitions[index].categorical_distribution.probabilities.front().probability;
  }

  return visualizePose(image, poseKeypoints);

}

//!
//! \brief detectPosesCallback ROS service call callback
//! \param req Service request
//! \param res Service response reference
//! \return True if succes, False otherwise
//!
bool detectPosesCallback(image_recognition_msgs::Recognize::Request& req, image_recognition_msgs::Recognize::Response& res)
{
  // ROS_INFO("detectPosesCallback");

  // Convert ROS message to opencv image
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(req.image, req.image.encoding);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("detectPosesCallback cv_bridge exception: %s", e.what());
    return false;
  }
  cv::Mat image = cv_ptr->image;
  if(image.empty())
  {
    ROS_ERROR("Empty image!");
    return false;
  }

  return detectPoses(image, res.recognitions);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "openpose");

  ros::NodeHandle local_nh("~");

  if (local_nh.hasParam("save_images_folder"))
  {
    g_save_images_folder = getParam(local_nh, "save_images_folder", std::string("/tmp"));
  }
  g_publish_to_topic = getParam(local_nh, "publish_result", true);

  op::Point<int> netInputSize = op::Point<int>(getParam(local_nh, "net_input_width", 656), getParam(local_nh, "net_input_height", 368));
  op::Point<int> outputSize = op::Point<int>(getParam(local_nh, "net_output_width", 656), getParam(local_nh, "net_output_height", 368));

  ros::Time start = ros::Time::now();
  g_openpose_wrapper = std::shared_ptr<OpenposeWrapper>(
        new OpenposeWrapper(netInputSize, outputSize,
                            getParam(local_nh, "num_scales", 1),
                            getParam(local_nh, "scale_gap", 0.3),
                            getParam(local_nh, "num_gpu_start", 0),
                            getParam(local_nh, "model_folder", std::string("/home/rasberry/git/openpose/models/")),
                            getParam(local_nh, "pose_model", std::string("BODY_25")),
                            getParam(local_nh, "overlay_alpha", 0.6)));
  ROS_INFO("OpenposeWrapper initialization took %.3f seconds", (ros::Time::now() - start).toSec());

  ros::NodeHandle nh;
  ros::ServiceServer service = nh.advertiseService("recognize", detectPosesCallback);
  ros::ServiceServer service2 = nh.advertiseService("visualize", visualizePoseCallback);
  if (g_publish_to_topic)
  {
    g_pub = nh.advertise<sensor_msgs::Image>("result_image", 1);
  }

  ros::spin();

  return 0;
}
