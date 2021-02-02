#ifndef FACIALLANDMARKDETECTOR_H
#define FACIALLANDMARKDETECTOR_H

#include "abstractlandmarkdetector.h"

#include <opencv2/core.hpp>

#include <string>
#include <stdexcept>
#include <vector>


//template <typename Landmarks>
class FacialLandmarkDetector : public AbstractLandmarkDetector<std::vector<cv::Point>>
//class FacialLandmarkDetector : public AbstractLandmarkDetector<Landmarks>
{
public:
	FacialLandmarkDetector(double scalingFactor)
		: scalingFactor(scalingFactor > 0 ? scalingFactor : throw std::invalid_argument("The scaling factor must be positive."))
		//, path("shape_predictor_68_face_landmarks.dat")
	{
	}

	// TODO: define copy/move semantics

	// TODO: add a getter/setter for the scaling factor

protected:

private:
	virtual std::unique_ptr<AbstractLandmarkDetector<std::vector<cv::Point>>> createClone() const override;
	//virtual std::unique_ptr<AbstractLandmarkDetector<Landmarks>> createClone() const override;

	virtual std::vector<cv::Point> detectLandmarks(const cv::Mat& image) const override;
	//virtual Landmarks detectLandmarks(const cv::Mat& image) const override;

	double scalingFactor = 1.0;
	static inline const std::string modelPath = "shape_predictor_68_face_landmarks.dat";
};	// FacialLandmarkDetector

/*
template <typename Landmarks>
std::unique_ptr<AbstractLandmarkDetector<Landmarks>> FacialLandmarkDetector<Landmarks>::createClone() const 
{
	//return std::make_unique<FacialLandmarkDetector>
	return std::unique_ptr<FacialLandmarkDetector<Landmarks>>(new FacialLandmarkDetector<Landmarks>(*this));
}

template <typename Landmarks>
Landmarks FacialLandmarkDetector<Landmarks>::detectLandmarks(const cv::Mat& image) const
{
	//return Landmarks();
	static_assert(false, "Landmark detection is not implemented for this type.");
}

template <>
inline std::vector<cv::Point> FacialLandmarkDetector<std::vector<cv::Point>>::detectLandmarks(const cv::Mat &image) const
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
			static_cast<long>(faces[0].left()*1.0/this->scalingFactor),
			static_cast<long>(faces[0].top()*1.0/this->scalingFactor),
			static_cast<long>(faces[0].right()*1.0/this->scalingFactor),
			static_cast<long>(faces[0].bottom()*1.0/this->scalingFactor));

		dlib::shape_predictor landmarkDetector;
		dlib::deserialize(this->modelPath) >> landmarkDetector;
				
		dlib::full_object_detection detection = landmarkDetector(dlib::cv_image<dlib::bgr_pixel>(image), faceRect);
		
		landmarks.reserve(detection.num_parts());
		for (unsigned long i = 0; i < detection.num_parts(); ++i)
		{
			const auto &p = detection.part(i);
			landmarks.emplace_back(p.x(), p.y());
		}
	}

	return landmarks;
}	// detectLandmarks
*/

#endif	// FACIALLANDMARKDETECTOR_H