#ifndef FACIALLANDMARKFILTER_H
#define FACIALLANDMARKFILTER_H

#include "abstractimagefilter.h"
#include "faciallandmarkdetector.h"

#include <opencv2/core.hpp>



// An abstract parent class for image filters based on facial landmarks

class FacialLandmarkFilter : public AbstractImageFilter
{
public:
	cv::Mat apply(const cv::Mat& image, const std::vector<cv::Point>& landmarks) const;

	cv::Mat apply(const cv::Mat& image, std::vector<cv::Point>&& landmarks) const;

	void apply(const cv::Mat& image, const std::vector<cv::Point>& landmarks, cv::Mat& out) const;

	void apply(const cv::Mat& image, std::vector<cv::Point>&& landmarks, cv::Mat& out) const;

	void applyInPlace(cv::Mat& image, const std::vector<cv::Point>& landmarks) const;

	void applyInPlace(cv::Mat& image, std::vector<cv::Point>&& landmarks) const;

	// prevent hiding inherited overloads of apply() and applyInPlace()
	using AbstractImageFilter::apply;	
	using AbstractImageFilter::applyInPlace;

protected:
	FacialLandmarkFilter(std::shared_ptr<FacialLandmarkDetector> landmarkDetector)
		: landmarkDetector(landmarkDetector) {}

	// TODO: define copy/move semantics

	// This function destructively reads existing landmark vector. In case there are no landmarks provided, they will be detected.
	// It is guaranteed that the member landmark vector will be left empty even if an exception is thrown.
	std::vector<cv::Point> grabLandmarks(const cv::Mat& image) const;

private:

	// This class helps to ensure that we will never reuse landmarks from a previous apply() call 
	struct ExistingLandmarksRAII
	{
		ExistingLandmarksRAII(std::vector<cv::Point>& landmarks)
			: landmarks(landmarks) {}

		~ExistingLandmarksRAII()
		{
			this->landmarks.clear();	// noexcept
		}

		std::vector<cv::Point>& landmarks;
	};	// ExistingLandmarksRAII


	virtual void modify(cv::Mat &image) const override = 0;

	std::shared_ptr<FacialLandmarkDetector> landmarkDetector;
	mutable std::vector<cv::Point> landmarks;
};	// FacialLandmarkFilter

#endif	// FACIALLANDMARKFILTER_H