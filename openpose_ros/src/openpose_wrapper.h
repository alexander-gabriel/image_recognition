#ifndef OPENPOSE_WRAPPER_H
#define OPENPOSE_WRAPPER_H

#include <cv_bridge/cv_bridge.h>
#include <openpose/headers.hpp>
#include <image_recognition_msgs/Recognition.h>

//! Forward declare openpose
namespace op
{
  class PoseExtractor;
  class PoseRenderer;
}

class OpenposeWrapper
{
public:
  //!
  //! \brief OpenposeWrapper Wraps the openpose implementation (this way we can use a dummy in simulation)
  //! \param net_input_size Input size of the network
  //! \param net_output_size Network output size
  //! \param scale_number Number of scales to average
  //! \param scale_gap Scale gap between scales
  //! \param num_gpu_start GPU device start number
  //! \param model_folder Where to find the openpose models
  //! \param pose_model Pose model string
  //! \param overlay_alpha Alpha factor used for overlaying the image
  //!
  OpenposeWrapper(const op::Point<int>& netInputSize, const op::Point<int>& outputSize, int scale_number,
                  double scale_gap, int num_gpu_start, const std::string &model_folder,
                  const std::string& poseModel_, double alpha_pose);

  bool detectPoses(const cv::Mat& image, std::vector<image_recognition_msgs::Recognition>& recognitions, cv::Mat& overlayed_image);
  bool visualizePose(const cv::Mat& inputImage, op::Array<float> poseKeypoints, cv::Mat& outputImage);

private:
  std::shared_ptr<op::PoseGpuRenderer> poseGpuRenderer;
  std::shared_ptr<op::PoseExtractorCaffe> poseExtractorPtr;
  op::ScaleAndSizeExtractor scaleAndSizeExtractor;
  op::CvMatToOpInput cvMatToOpInput;
  op::CvMatToOpOutput cvMatToOpOutput;
  op::OpOutputToCvMat opOutputToCvMat;
  std::map<unsigned int, std::string> bodypart_map_;
  op::PoseModel poseModel;
  op::Point<int> netInputSize;
  op::Point<int> outputSize;

  int scale_number;
  double scale_gap;

};

#endif // OPENPOSE_WRAPPER_H
