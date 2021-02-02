#include "faciallandmarkdetector.h"

#include <opencv2/imgproc.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>


std::unique_ptr<AbstractLandmarkDetector<std::vector<cv::Point>>> FacialLandmarkDetector::createClone() const
{
	return std::unique_ptr<AbstractLandmarkDetector<std::vector<cv::Point>>>(new FacialLandmarkDetector(*this));
}

std::vector<cv::Point> FacialLandmarkDetector::detectLandmarks(const cv::Mat& image) const
{
	CV_Assert(!image.empty());

	std::vector<cv::Point> landmarks;

	cv::Mat imageSmall;
	cv::resize(image, imageSmall, cv::Size(), this->scalingFactor, this->scalingFactor);

	dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

	auto faces = faceDetector(dlib::cv_image<dlib::bgr_pixel>(imageSmall));

	if (faces.size() < 1)
	{
		throw std::runtime_error("Failed to detect a face in the provided image.");
	}
	else
	{
		dlib::rectangle faceRect(
			static_cast<long>(faces[0].left() * 1.0 / this->scalingFactor),
			static_cast<long>(faces[0].top() * 1.0 / this->scalingFactor),
			static_cast<long>(faces[0].right() * 1.0 / this->scalingFactor),
			static_cast<long>(faces[0].bottom() * 1.0 / this->scalingFactor));

		dlib::shape_predictor landmarkDetector;
		dlib::deserialize(this->modelPath) >> landmarkDetector;

		dlib::full_object_detection detection = landmarkDetector(dlib::cv_image<dlib::bgr_pixel>(image), faceRect);

		landmarks.reserve(detection.num_parts());
		for (unsigned long i = 0; i < detection.num_parts(); ++i)
		{
			const auto& p = detection.part(i);
			landmarks.emplace_back(p.x(), p.y());
		}
	}

	return landmarks;
}