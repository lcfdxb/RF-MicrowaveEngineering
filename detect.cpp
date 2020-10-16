/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2017, Kentaro Wada.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Kentaro Wada nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

// opencv and boost
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/hal/interface.h>
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

// ROS
#include <cv_bridge/cv_bridge.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include "vision_msgs/Detection2DArray.h"
#include "vision_msgs/Detection2D.h"
#include "vision_msgs/ObjectHypothesisWithPose.h"
#include "vision_msgs/VisionInfo.h"

// helper functions from qualcomm example
#include "snpe/CreateUserBuffer.hpp"

// snpe includes
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DiagLog/IDiagLog.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensorFactory.hpp"

namespace snpe_ros
{
class DetectNodelet : public nodelet::Nodelet
{
protected:
  // ROS communication
  image_transport::Subscriber sub_image_;
  image_transport::Publisher pub_rect_;
  ros::Publisher detections_pub_;
  ros::Publisher vision_info_pub_;

  boost::shared_ptr<image_transport::ImageTransport> it_;
  boost::mutex connect_mutex_;

  enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR};
  enum {CPUBUFFER, GLBUFFER};

  int queue_size_;

  // static std::string dlc;
  std::string runtime_target_;
  std::string bufferTypeStr;
  std::string userBufferSourceStr = "CPUBUFFER";
  zdl::DlSystem::Runtime_t runtime;
  zdl::DlSystem::RuntimeList runtimeList;
  bool runtimeSpecified = true;
  bool execStatus = false;
  bool usingInitCaching = false;
  bool crop_ = true;

  float mean_val_, scale_;
  float confidence_threshold_;

  std::string scoresName_;
  std::string classesName_;
  std::string boxesName_;

  size_t tensorWidth;
  size_t tensorHeight;
  size_t tensorChannels;
  size_t tensorLength;
  zdl::DlSystem::TensorShape tensorShape;

  vision_msgs::VisionInfo vision_info_msg_;

  std::unique_ptr<zdl::SNPE::SNPE> snpe;

  std::vector<std::string> classLabels_;
  std::vector<std::string> additionalOutputLayers_;

  virtual void onInit();
  void connectCb();
  void imageCb(const sensor_msgs::ImageConstPtr& image_msg);

};

void DetectNodelet::onInit()
{
  ros::NodeHandle &nh         = getNodeHandle();
  ros::NodeHandle &private_nh = getPrivateNodeHandle();
  it_.reset(new image_transport::ImageTransport(private_nh));

  std::string dlc_path;
  int database_version;
  runtime = zdl::DlSystem::Runtime_t::CPU;
  zdl::DlSystem::RuntimeList runtimeList;

  bool cpu_fallback_enable;

  // preferences for processing/output
  private_nh.param("crop",                  crop_,                  true);
  private_nh.param("confidence_threshold",  confidence_threshold_,  0.3f);
  private_nh.param<std::string>("runtime_target", runtime_target_, "gpu");

  // ROS parameters
  private_nh.param("queue_size", queue_size_, 5);

  // network specific values
  private_nh.param("cpu_fallback_enable", cpu_fallback_enable, true);
  private_nh.param("mean_val", mean_val_, 127.5f);
  private_nh.param("scale", scale_, 0.007843f);
  private_nh.param<std::string>("dlc",                    dlc_path,       "network.dlc");
  private_nh.param<std::string>("layer_names/scores",     scoresName_,    "Postprocessor/BatchMultiClassNonMaxSuppression_scores");
  private_nh.param<std::string>("layer_names/classes",    classesName_,   "detection_classes:0");
  private_nh.param<std::string>("layer_names/boxes",      boxesName_,     "Postprocessor/BatchMultiClassNonMaxSuppression_boxes");
  private_nh.param<std::string>("buffer_type",            bufferTypeStr,  "ITENSOR");
  private_nh.param("database_version",                    database_version,1);

  // read all labels
  private_nh.param("labels", classLabels_, std::vector<std::string>(0));

  // read any additional layers
  private_nh.param("additional_output_layers", additionalOutputLayers_, std::vector<std::string>(0));

  NODELET_DEBUG("dlc_path is: %s", dlc_path.c_str());
  NODELET_DEBUG("Number of labels: %i", classLabels_.size());
  NODELET_DEBUG("Number of additional_output_layers: %i", additionalOutputLayers_.size());

  for (int i = 0; i < classLabels_.size(); i++) {
    NODELET_DEBUG("Label %i: %s",i,classLabels_[i].c_str());
  }

  // craft vision info message (only sent on detection)
  vision_info_msg_.database_location = private_nh.getNamespace() + "/labels";
  vision_info_msg_.database_version = database_version;

  // determine runtime target
  if (runtime_target_.compare("gpu") == 0)
  {
      runtime = zdl::DlSystem::Runtime_t::GPU;
  }
  else if (runtime_target_.compare("aip") == 0)
  {
      runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
  }
  else if (runtime_target_.compare("dsp") == 0)
  {
      runtime = zdl::DlSystem::Runtime_t::DSP;
  }
  else if (runtime_target_.compare("cpu") == 0)
  {
     runtime = zdl::DlSystem::Runtime_t::CPU;
  }
  else
  {
     NODELET_WARN("The runtime option provide is not valid. Defaulting to the CPU runtime.");
     runtime = zdl::DlSystem::Runtime_t::CPU;
  }

  runtimeList.add(runtime);

  // some NN need CPU for some layers
  if (cpu_fallback_enable) {
    runtimeList.add(zdl::DlSystem::Runtime_t::CPU);
  }

  // Check if given buffer type is valid
  int bufferType;
  if (bufferTypeStr == "USERBUFFER_FLOAT")
  {
      bufferType = USERBUFFER_FLOAT;
  }
  else if (bufferTypeStr == "USERBUFFER_TF8")
  {
      bufferType = USERBUFFER_TF8;
  }
  else if (bufferTypeStr == "ITENSOR")
  {
      bufferType = ITENSOR;
  }
  else
  {
    NODELET_ERROR("Buffer type is not valid. Please run snpe-sample with the -h flag for more details");
  }

    //Check if given user buffer source type is valid
  int userBufferSourceType;
  // CPUBUFFER / GLBUFFER supported only for USERBUFFER_FLOAT
  if (bufferType == USERBUFFER_FLOAT)
  {
      if( userBufferSourceStr == "CPUBUFFER" )
      {
          userBufferSourceType = CPUBUFFER;
      }
      else if( userBufferSourceStr == "GLBUFFER" )
      {
          userBufferSourceType = GLBUFFER;
      }
      else
      {
        NODELET_ERROR("Source of user buffer type is not valid. Please run snpe-sample with the -h flag for more details");
      }
  }

  if(runtimeSpecified)
  {
      static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
      std::cout << "SNPE Version: " << Version.asString().c_str() << std::endl; //Print Version number
  }

  std::unique_ptr<zdl::DlContainer::IDlContainer> container = 
    zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlc_path.c_str()));
  if (container == nullptr)
  {
     NODELET_ERROR("Error while opening the container file.");
  }

  bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT || bufferType == USERBUFFER_TF8);

  zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

  // amalyshe: this is standard names of the layers for models been crfeated
  // using Tensor Flow object detection API (at least i TF 1.xx versions)
  // let's stick to these names if we do not enumerate these names explicitly,
  // there will be only one explicit output responsible for classes adding of
  // second layer gives us three more buffers which will have boxes and
  // scores
  zdl::DlSystem::StringList outputs;

  // if there are additional output layers
  if  (additionalOutputLayers_.size()) {
    for (int i = 0; i < additionalOutputLayers_.size(); i++) {
        outputs.append("add");
        outputs.append(additionalOutputLayers_[i].c_str());
    }
  }

  zdl::DlSystem::PlatformConfig platformConfig;
  snpe = snpeBuilder.setOutputLayers(outputs)
       .setRuntimeProcessorOrder(runtimeList)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .setInitCacheMode(usingInitCaching)
       .build();
  if (snpe == nullptr)
  {
     NODELET_ERROR("Error while building SNPE object.");
  }

  // Check the batch size for the container
  // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
  // is the batch size.
  tensorShape = snpe->getInputDimensions();
  size_t batchSize = tensorShape.getDimensions()[0];
  tensorWidth = tensorShape.getDimensions()[1];
  tensorHeight = tensorShape.getDimensions()[2];
  tensorChannels = tensorShape.getDimensions()[3];
  tensorLength = tensorWidth * tensorHeight * tensorChannels * batchSize;

  NODELET_INFO("Batch size for the container is %i", (int)batchSize);
  NODELET_INFO("Width for the container is %i", (int)tensorWidth);
  NODELET_INFO("Height for the container is %i", (int)tensorHeight);
  NODELET_INFO("Channels for the container is %i", (int)tensorChannels);

  // Monitor whether anyone is subscribed to the output
  image_transport::SubscriberStatusCallback connect_cb = boost::bind(&DetectNodelet::connectCb, this);
  // Make sure we don't enter connectCb() between advertising and assigning to
  // pub_XXX
  boost::lock_guard<boost::mutex> lock(connect_mutex_);
  pub_rect_  = it_->advertise("detections_image",  1, connect_cb, connect_cb);

  // publishers
  ros::SubscriberStatusCallback det_connect_cb = boost::bind(&DetectNodelet::connectCb, this);
  detections_pub_   = private_nh.advertise<vision_msgs::Detection2DArray>("detections", 1, det_connect_cb, det_connect_cb);
  vision_info_pub_  = private_nh.advertise<vision_msgs::VisionInfo>("vision_info", 1);

}

// Handles (un)subscribing when clients (un)subscribe
void DetectNodelet::connectCb()
{
  boost::lock_guard<boost::mutex> lock(connect_mutex_);
  if (pub_rect_.getNumSubscribers() == 0 && detections_pub_.getNumSubscribers() == 0)
    sub_image_.shutdown();
  else if (!sub_image_)
  {
    image_transport::TransportHints hints("raw", ros::TransportHints(), getPrivateNodeHandle());
    sub_image_ = it_->subscribe("image_raw", queue_size_, &DetectNodelet::imageCb, this, hints);
  }
}

void DetectNodelet::imageCb(const sensor_msgs::ImageConstPtr& image_msg)
{
  boost::posix_time::ptime rT1, rT2, rT3, rT4;
  rT1 =  boost::posix_time::microsec_clock::local_time();

  cv::Mat image = cv_bridge::toCvShare(image_msg)->image;

  // Need to get image in NHWC format - Batch #, Height, Width, Channels
  // Tensor likely has fixed input dimensions

  // crop if necessary and requested
  cv::Mat crop;
  int padding_x = 0;
  int padding_y = 0;
  if (image_msg->width != image_msg->height && crop_) {
    cv::Rect roi;

    if (image_msg->width > image_msg->height) {
      roi.x = (image.size().width - image.size().height)/2;
      roi.y = 0;
      roi.width = image.size().height;
      roi.height = image.size().height;
    } else {
      roi.x = 0;
      roi.y = (image.size().height - image.size().width)/2;
      roi.width = image.size().width;
      roi.height = image.size().width;
    }

    crop = image(roi);
    padding_x = roi.x;
    padding_y = roi.y;
  } else {
    crop = image;
  }

  // // resize image if necessary
  cv::Mat scaled_cv;
  if(crop.size().width != tensorWidth || crop.size().height != tensorHeight) {
    cv::resize(crop, scaled_cv, cv::Size((int)tensorWidth, (int)tensorHeight), 0, 0);
    // scale_x = crop.size().width / scaled_cv.size().width;
    // scale_y = crop.size().height / scaled_cv.size().height;
  } else {
    scaled_cv = crop;
  }

  // if grayscale convert to RGB
  cv::Mat rgb_cv;
  if (sensor_msgs::image_encodings::isMono(image_msg->encoding)) {
    cv::cvtColor(scaled_cv, rgb_cv, cv::COLOR_GRAY2RGB);
  } else if (sensor_msgs::image_encodings::isColor(image_msg->encoding)) {
    rgb_cv = scaled_cv;
  }

  // rotate image
  cv::Mat resized_image(rgb_cv);
  cv::resize(rgb_cv, resized_image, cv::Size(tensorHeight, tensorWidth));

  // size_t imageLength = inputBlob_NCWH.size[0] * inputBlob_NCWH.size[1] * inputBlob_NCWH.size[2] * inputBlob_NCWH.size[3];
  size_t imageLength = resized_image.channels() * resized_image.cols * resized_image.rows;

  std::unique_ptr<zdl::DlSystem::ITensor> input_tensor =
      zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(snpe->getInputDimensions());

  zdl::DlSystem::ITensor *t = input_tensor.get();
  if (!input_tensor.get()) {
      std::cerr << "Could not create SNPE input tensor" << std::endl;
      return;
  }


  // // Follow copying of image is not optimal since it happens in two passes.
  // Also applies mean value and scales
  float *tf = reinterpret_cast<float *>(&(*input_tensor->begin()));
  for (size_t i = 0; i < imageLength; i++) {
    tf[i] = (static_cast<float>(resized_image.data[i]) - mean_val_ ) * scale_;
  }

  // reorder from BGR to RGB:
  // snpe-tensorflow-to-dlc has follow parameters --input_encoding "input" bgr --input_type "input" image
  // unfortunately they do not work
  // if they worked, I would not have such code here
  if (image_msg->encoding == sensor_msgs::image_encodings::BGR8) {
    for (size_t i = 0; i < tensorLength; i += tensorChannels) {
        float tmp = tf[i];
        tf[i] = tf[i + 2];
        tf[i + 2] = tmp;
    }
  }

  // setup output
  zdl::DlSystem::TensorMap outputTensorMap;
  zdl::DlSystem::UserBufferMap inputMap, outputMap;
  std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
  std::unordered_map <std::string, std::vector<uint8_t>> applicationOutputBuffers;
  createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, false);

  rT2 =  boost::posix_time::microsec_clock::local_time();

  // inference
  // execStatus = snpe->execute(input_tensor.get(), outputTensorMap);
  execStatus = snpe->execute(t, outputTensorMap);

  // return error if something goes wrong
  if (!execStatus)
  {
    const char* const errStr = zdl::DlSystem::getLastErrorString();
    NODELET_ERROR("snpe_ros inference error: %s",errStr);
    return;
  }

  rT3 =  boost::posix_time::microsec_clock::local_time();

  // // Get all output tensor names from the network
  zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();

  // mode debugging output
  // for (size_t i = 0; i < tensorNames.size(); ++i) {
  //   std::string output_name(tensorNames.at(i));

  //   NODELET_INFO("tensor output %i: %s", i, output_name.c_str());
  // }

  // debugging output
  // for (size_t i = 0; i < tensorNames.size(); i++) {
  //   std::cout << tensorNames.at(i) << " (";
  //   zdl::DlSystem::ITensor *outTensori = outputTensorMap.getTensor(tensorNames.at(i));
  //   zdl::DlSystem::TensorShape shapei = outTensori->getShape();
  //   for (size_t j = 0; j < shapei.rank(); j++) {
  //       std::cout << shapei[j] << ", ";
  //   }
  //   std::cout  << ")" << std::endl;
  // }

  if (tensorNames.size() != 4) {
      std::cerr << "Output list has " << tensorNames.size() << " elements, while three are expected" << std::endl;
      NODELET_ERROR("Output list has %i elements, while three are expected",tensorNames.size());
      return;
  }

  // Looking for the Top5 values and print them in format
  // class_id     probability/value of the latest tensor in case of softmax absence
  zdl::DlSystem::ITensor *outTensorScores = outputTensorMap.getTensor(scoresName_.c_str());
  zdl::DlSystem::ITensor *outTensorClasses = outputTensorMap.getTensor(classesName_.c_str());
  zdl::DlSystem::ITensor *outTensorBoxes = outputTensorMap.getTensor(boxesName_.c_str());
  zdl::DlSystem::TensorShape scoresShape = outTensorScores->getShape();
  if (scoresShape.rank() != 2) {
      std::cerr << "Scores should have two axis" << std::endl;
      return;
  }

  const float *oScores = reinterpret_cast<float *>(&(*outTensorScores->begin()));
  const float *oClasses = reinterpret_cast<float *>(&(*outTensorClasses->begin()));
  const float *oBoxes = reinterpret_cast<float *>(&(*outTensorBoxes->begin()));

  std::vector<int> boxes;
  std::vector<int> classes;

  vision_msgs::Detection2DArray detections;
  detections.header = image_msg->header;

  cv::Mat detectionMat = image;

  for (size_t curProposal = 0; curProposal < scoresShape[1]; curProposal++) {
    float confidence = oScores[curProposal];
    float label_float = static_cast<int>(oClasses[curProposal]);
    std::string label = classLabels_[(int)label_float];

    if (confidence > confidence_threshold_) {

      // boxes have follow layout top, left, bottom, right
      // according to this link: https://www.tensorflow.org/lite/models/object_detection/overview
      auto yLeftBottom  = padding_y + static_cast<int>(oBoxes[4 * curProposal] * crop.size().height);
      auto xLeftBottom  = padding_x + static_cast<int>(oBoxes[4 * curProposal + 1] * crop.size().width);
      auto yRightTop    = padding_y + static_cast<int>(oBoxes[4 * curProposal + 2] * crop.size().height);
      auto xRightTop    = padding_x + static_cast<int>(oBoxes[4 * curProposal + 3] * crop.size().width);

      // draw box
      cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                      (int)(xRightTop - xLeftBottom),
                      (int)(yRightTop - yLeftBottom));
      cv::rectangle(detectionMat, object, cv::Scalar(0, 255, 0));

      // draw label
      cv::String label_string = cv::String(label);
      int baseLine = 0;
      cv::Size labelSize = getTextSize(label_string, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      cv::rectangle(detectionMat, 
        cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height),
        cv::Size(labelSize.width, labelSize.height + baseLine)),
        cv::Scalar(255, 255, 255), 
        CV_FILLED);
      putText(detectionMat, 
        label, 
        cv::Point(xLeftBottom, yLeftBottom),
        cv::FONT_HERSHEY_SIMPLEX, 
        0.5, 
        cv::Scalar(0,0,0));

      // save bounding box
      vision_msgs::Detection2D detection;
      detection.bbox.center.x = (xLeftBottom + xRightTop)/2.0f;
      detection.bbox.center.y = (yLeftBottom + yRightTop)/2.0f;
      detection.bbox.size_x = xRightTop - xLeftBottom;
      detection.bbox.size_y = yRightTop - yLeftBottom;

      vision_msgs::ObjectHypothesisWithPose hypo;
      hypo.id = (int)label_float;
      hypo.score = confidence;
      detection.results.push_back(hypo);
      detections.detections.push_back(detection);

      NODELET_DEBUG("[%u, %s] element, prob = %3.3f", curProposal, label.c_str(), confidence);

      // std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence <<
      // "    (" << xLeftBottom << "," << yLeftBottom << ")-(" << xRightTop << "," << yRightTop << ")" << std::endl;
    }
  }

  // output image with boxes
  sensor_msgs::ImagePtr detect_msg = 
    cv_bridge::CvImage(image_msg->header, image_msg->encoding, detectionMat).toImageMsg();
  pub_rect_.publish(detect_msg);

  // output detections
  detections_pub_.publish(detections);

  // output vision message
  vision_info_msg_.header = image_msg->header;
  vision_info_pub_.publish(vision_info_msg_);

  rT4 =  boost::posix_time::microsec_clock::local_time();

  NODELET_DEBUG("%.4f seconds to preprocess\n",(rT2-rT1).total_microseconds() * 1e-6);
  NODELET_DEBUG("%.4f seconds to inference\n",(rT3-rT2).total_microseconds() * 1e-6);
  NODELET_DEBUG("%.4f seconds to postprocess\n",(rT4-rT3).total_microseconds() * 1e-6);


}

}  // namespace snpe_ros

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(snpe_ros::DetectNodelet, nodelet::Nodelet)
