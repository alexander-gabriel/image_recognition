#include "openpose_wrapper.h"

#include <ros/console.h>
#include <gflags/gflags.h>
#include <openpose/headers.hpp>

DEFINE_bool(disable_blending,           false,          "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
" background, instead of being rendered into the original image. Related: `part_to_show`,"
" `alpha_pose`, and `alpha_pose`.");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
" hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
" heatmap, 0 will only show the frame. Only valid for GPU rendering.");
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
" rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
" while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
" more false positives (i.e. wrong detections).");
DEFINE_int32(part_to_show,              0,             "Prediction channel to visualize (default: 0). 0 for all the body parts, 1-18 for each body"
" part heat map, 19 for the background heat map, 20 for all the body part heat maps"
" together, 21 for all the PAFs, 22-40 for each body part pair PAF.");
DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will fill op::Datum::poseHeatMaps array with the body part heatmaps, and"
" analogously face & hand heatmaps to op::Datum::faceHeatMaps & op::Datum::handHeatMaps."
" If more than one `add_heatmaps_X` flag is enabled, it will place then in sequential"
" memory order: body parts + bkg + PAFs. It will follow the order on"
" POSE_BODY_PART_MAPPING in `src/openpose/pose/poseParameters.cpp`. Program speed will"
" considerably decrease. Not required for OpenPose, enable it only if you intend to"
" explicitly use this information later.");
DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to"
" background.");
DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
DEFINE_int32(heatmaps_scale,            2,              "Set 0 to scale op::Datum::poseHeatMaps in the range [-1,1], 1 for [0,1]; 2 for integer"
" rounded [0,255]; and 3 for no scaling.");
DEFINE_bool(part_candidates,            false,          "Also enable `write_json` in order to save this information. If true, it will fill the"
" op::Datum::poseCandidates array with the body part candidates. Candidates refer to all"
" the detected body parts, before being assembled into people. Note that the number of"
" candidates is equal or higher than the number of final body parts (i.e. after being"
" assembled into people). The empty body parts are filled with 0s. Program speed will"
" slightly decrease. Not required for OpenPose, enable it only if you intend to explicitly"
" use this information.");
DEFINE_string(model_pose,               "COCO",      "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
"`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder,             "/home/alex/git/openpose/models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
" input image resolution.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
" If you want to change the initial scale, you actually want to multiply the"
" `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");

OpenposeWrapper::OpenposeWrapper(const op::Point<int>& netInputSize, const op::Point<int>& outputSize,
  int scale_number, double scale_gap,
  int num_gpu_start, const std::string& model_folder,
  const std::string& poseModel_, double alpha_pose) :
  netInputSize(netInputSize),
  outputSize(outputSize),
  scale_number(scale_number),
  scale_gap(scale_gap),
  scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap),
  cvMatToOpInput{op::flagsToPoseModel(poseModel_)},
  poseModel(op::flagsToPoseModel(poseModel_)),
  bodypart_map_(getPoseBodyPartMapping(op::flagsToPoseModel(poseModel_)))
  {

    // ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
    op::ConfigureLog::setPriorityThreshold(op::Priority::High);

    // Logging
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);

    // Step 3 - Initialize all required classes


    poseExtractorPtr = std::make_shared<op::PoseExtractorCaffe>(poseModel, model_folder, 0);
    poseGpuRenderer = std::shared_ptr<op::PoseGpuRenderer>(new op::PoseGpuRenderer{poseModel, poseExtractorPtr, (float)FLAGS_render_threshold, !FLAGS_disable_blending, (float)alpha_pose, (float)FLAGS_alpha_heatmap});
    poseGpuRenderer->setElementToRender(FLAGS_part_to_show);

    op::FrameDisplayer frameDisplayer{"OpenPose Tutorial - Example 2", outputSize};
    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseExtractorPtr->initializationOnThread();
    poseGpuRenderer->initializationOnThread();
    // const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
    // const auto heatMapScale = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
    //
    // op::Point<int> op_netInputSize(netInputSize_.width, netInputSize_.height);
    // op::Point<int> op_net_output_size(net_output_size_.width, net_output_size_.height);
    //
    // ROS_INFO("Net input size: [%d x %d]", op_netInputSize.x, op_netInputSize.y);
    // ROS_INFO("Net output size: [%d x %d]", op_net_output_size.x, op_net_output_size.y);


  }


  bool OpenposeWrapper::visualizePose(const cv::Mat& inputImage, Array<float> poseKeypoints, cv::Mat& outputImage) {

    const op::Point<int> imageSize{inputImage.cols, inputImage.rows};
    ROS_INFO("Step 2");
    // Step 2 - Get desired scale sizes
    std::vector<double> scaleInputToNetInputs;
    std::vector<op::Point<int>> netInputSizes;
    double scaleInputToOutput;
    op::Point<int> outputResolution;
    std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
    = scaleAndSizeExtractor.extract(imageSize);
    const auto scaleNetToOutput = poseExtractorPtr->getScaleNetToOutput();

    auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
    // Step 5 - Render pose
    poseGpuRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput, scaleNetToOutput);
    // Step 6 - OpenPose output format to cv::Mat
    outputImage = opOutputToCvMat.formatToCvMat(outputArray);

    // Calculate the factors between the input image and the output image
    double width_factor = (double) inputImage.cols / outputImage.cols;
    double height_factor = (double) inputImage.rows / outputImage.rows;
    double scale_factor = std::fmax(width_factor, height_factor);

    recognitions.resize(num_people * num_bodyparts);
  }

  bool OpenposeWrapper::detectPoses(const cv::Mat& inputImage, std::vector<image_recognition_msgs::Recognition>& recognitions, cv::Mat& outputImage)
  {
    ROS_INFO("OpenposeWrapper::detectPoses: Detecting poses on image of size [%d x %d]", inputImage.cols, inputImage.rows);

    const op::Point<int> imageSize{inputImage.cols, inputImage.rows};
    // Step 2 - Get desired scale sizes
    std::vector<double> scaleInputToNetInputs;
    std::vector<op::Point<int>> netInputSizes;
    double scaleInputToOutput;
    op::Point<int> outputResolution;
    std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
    = scaleAndSizeExtractor.extract(imageSize);
    // Step 3 - Format input image to OpenPose input and output formats
    const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
    auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
    // Step 4 - Estimate poseKeypoints

    //EXTRA
    // op::PoseExtractorCaffe poseExtractorCaffe{poseModel, "/home/alex/git/openpose/models/", 0};
    // op::PoseCpuRenderer poseRenderer{poseModel, (float)0.05, true, (float)0.6};
    // poseExtractorCaffe.initializationOnThread();
    // poseRenderer.initializationOnThread();
    // poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
    // const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();
    //EXTRA

    poseExtractorPtr->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
    const auto poseKeypoints = poseExtractorPtr->getPoseKeypoints();
    const auto scaleNetToOutput = poseExtractorPtr->getScaleNetToOutput();

    size_t num_people = poseKeypoints.getSize(0);
    size_t num_bodyparts = poseKeypoints.getSize(1);
    ROS_INFO("OpenposeWrapper::detectPoses: Rendering %d keypoints", (int) (num_people * num_bodyparts));
    // Step 5 - Render pose
    //TODO: this segfaults for whatever reason..memory?
    poseGpuRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput, scaleNetToOutput);
    // Step 6 - OpenPose output format to cv::Mat
    outputImage = opOutputToCvMat.formatToCvMat(outputArray);
    // Calculate the factors between the input image and the output image
    double width_factor = (double) inputImage.cols / outputImage.cols;
    double height_factor = (double) inputImage.rows / outputImage.rows;
    double scale_factor = std::fmax(width_factor, height_factor);
    recognitions.resize(num_people * num_bodyparts);

    ROS_INFO("OpenposeWrapper::detectPoses: Detected %d persons", (int) num_people);

    for (size_t person_idx = 0; person_idx < num_people; person_idx++)
    {
      for (size_t bodypart_idx = 0; bodypart_idx < num_bodyparts; bodypart_idx++)
      {
        size_t index = (person_idx * num_bodyparts + bodypart_idx);

        recognitions[index].group_id = person_idx;
        recognitions[index].roi.width = 1;
        recognitions[index].roi.height = 1;
        recognitions[index].categorical_distribution.probabilities.resize(1);
        recognitions[index].categorical_distribution.probabilities.front().label = bodypart_map_[bodypart_idx];

        recognitions[index].roi.x_offset = poseKeypoints[3 * index] * scale_factor;
        recognitions[index].roi.y_offset = poseKeypoints[3 * index + 1] * scale_factor;
        recognitions[index].categorical_distribution.probabilities.front().probability = poseKeypoints[3 * index + 2];
      }
    }

    return true;



    //   const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
    //   const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    //   op::Point<int> op_netInputSize(netInputSize_.width, netInputSize_.height);
    //   op::CvMatToOpInput cvMatToOpInput{poseModel};
    //   std::vector<double> scaleInputToNetInputs;
    //   std::vector<op::Point<int>> netInputSizes;
    //   double scaleInputToOutput;
    //   op::Point<int> outputResolution;
    //   op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize_, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
    //   std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
    // = scaleAndSizeExtractor.extract(op_netInputSize);
    //
    //   const auto netInputArray = cvMatToOpInput.createArray(image, scaleInputToNetInputs, netInputSizes);
    //
    //
    //   std::vector<float> scale_ratios;
    //   std::tie(net_input_array, scale_ratios) = cv_mat_to_input.format(image);
    //
    //   ROS_INFO("OpenposeWrapper::detectPoses: Net input size: [%d x %d]", op_netInputSize.x, op_netInputSize.y);
    //
    //   op::Point<int> op_net_output_size(net_output_size_.width, net_output_size_.height);
    //   op::OpOutputToCvMat op_output_to_cv_mat(op_net_output_size);
    //   op::CvMatToOpOutput cv_mat_to_output(op_net_output_size);
    //
    //   ROS_INFO("OpenposeWrapper::detectPoses: Net output size: [%d x %d]", op_net_output_size.x, op_net_output_size.y);
    //
    //   op::Array<float> output_array;
    //   double scale_input_to_output;
    //   std::tie(scale_input_to_output, output_array) = cv_mat_to_output.format(image);
    //
    //   ROS_INFO("OpenposeWrapper::detectPoses: Applying forward pass on image of size: [%d x %d]", image.cols, image.rows);
    //
    //   // Step 3 - Estimate poseKeyPoints
    //   pose_extractor_->forwardPass(net_input_array, {image.cols, image.rows}, scale_ratios);
    //   const auto pose_keypoints = pose_extractor_->getPoseKeypoints();
    //
    //   size_t num_people = pose_keypoints.getSize(0);
    //   size_t num_bodyparts = pose_keypoints.getSize(1);
    //
    //   ROS_INFO("OpenposeWrapper::detectPoses: Rendering %d keypoints", (int) (num_people * num_bodyparts));
    //
    //   // Step 4 - Render poseKeyPoints
    //   pose_renderer_->renderPose(output_array, pose_keypoints);
    //
    //   // Step 5 - OpenPose output format to cv::Mat
    //   overlayed_image = op_output_to_cv_mat.formatToCvMat(output_array);
    //
    //   // Calculate the factors between the input image and the output image
    //   double width_factor = (double) image.cols / overlayed_image.cols;
    //   double height_factor = (double) image.rows / overlayed_image.rows;
    //   double scale_factor = std::fmax(width_factor, height_factor);
    //
    //   recognitions.resize(num_people * num_bodyparts);
    //
    //   ROS_INFO("OpenposeWrapper::detectPoses: Detected %d persons", (int) num_people);
    //
    //   for (size_t person_idx = 0; person_idx < num_people; person_idx++)
    //   {
    //     for (size_t bodypart_idx = 0; bodypart_idx < num_bodyparts; bodypart_idx++)
    //     {
    //       size_t index = (person_idx * num_bodyparts + bodypart_idx);
    //
    //       recognitions[index].group_id = person_idx;
    //       recognitions[index].roi.width = 1;
    //       recognitions[index].roi.height = 1;
    //       recognitions[index].categorical_distribution.probabilities.resize(1);
    //       recognitions[index].categorical_distribution.probabilities.front().label = bodypart_map_[bodypart_idx];
    //
    //       recognitions[index].roi.x_offset = pose_keypoints[3 * index] * scale_factor;
    //       recognitions[index].roi.y_offset = pose_keypoints[3 * index + 1] * scale_factor;
    //       recognitions[index].categorical_distribution.probabilities.front().probability = pose_keypoints[3 * index + 2];
    //     }
    //   }
    //
    //   return true;
  }
