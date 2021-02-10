#ifndef FACIALLANDMARKDETECTOR_H
#define FACIALLANDMARKDETECTOR_H

#include "abstractlandmarkdetector.h"

#include <opencv2/core.hpp>

#include <string>
#include <stdexcept>
#include <vector>


class FacialLandmarkDetector : public AbstractLandmarkDetector<std::vector<cv::Point>>
{
public:

	constexpr FacialLandmarkDetector(double scalingFactor)
		: scalingFactor(scalingFactor > 0 ? scalingFactor : throw std::invalid_argument("The scaling factor must be positive."))
	{
	}


	// The scaling factor is used for resizing the image to speed up face detection. Therefore, it should normally be less than 1.
	
	constexpr double getScalingFactor() const noexcept { return this->scalingFactor; }

	constexpr void setScalingFactor(double scalingFactor) 
	{
		this->scalingFactor = scalingFactor > 0 ? scalingFactor : throw std::invalid_argument("The scaling factor must be positive.");
	}

protected:	

	FacialLandmarkDetector(const FacialLandmarkDetector&) = default;
	FacialLandmarkDetector(FacialLandmarkDetector&&) = default;

	FacialLandmarkDetector& operator = (const FacialLandmarkDetector&) = delete;
	FacialLandmarkDetector& operator = (FacialLandmarkDetector&&) = delete;

private:

	virtual std::unique_ptr<AbstractLandmarkDetector<std::vector<cv::Point>>> createClone() const override;

	virtual std::vector<cv::Point> detectLandmarks(const cv::Mat& image) const override;

	double scalingFactor = 1.0;

	static inline const std::string modelPath = "shape_predictor_68_face_landmarks.dat";
};	// FacialLandmarkDetector


#endif	// FACIALLANDMARKDETECTOR_H