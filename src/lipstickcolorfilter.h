#ifndef LIPSTICKCOLORFILTER_H
#define LIPSTICKCOLORFILTER_H

//#include "abstractimagefilter.h"
#include "faciallandmarkfilter.h"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>


//class LipstickColorFilter : public AbstractImageFilter
class LipstickColorFilter : public FacialLandmarkFilter
{
public:
	LipstickColorFilter(std::shared_ptr<FacialLandmarkDetector> landmarkDetector)
		: FacialLandmarkFilter(std::move(landmarkDetector)) {}

	// TODO: define copy/move semantics

	// cv::Scalar's copy constructor is not noexcept
	cv::Scalar getColor() const /*noexcept*/ { return this->color; }

	void setColor(const cv::Scalar& rgba) { this->color = rgba; }

	//void setLandmarks(const std::vector<cv::Point>& landmarks) { this->landmarks = landmarks; }
	//void setLandmarks(std::vector<cv::Point>&& landmarks) { this->landmarks = std::move(landmarks); }

	//cv::Mat apply(const cv::Mat& image, const std::vector<cv::Point> &landmarks) const;

	//void apply(const cv::Mat& image, const std::vector<cv::Point>& landmarks, cv::Mat& out) const;

	//// Prevent new apply() overloads from hiding inherited ones
	//using AbstractImageFilter::apply;

private:

	virtual std::unique_ptr<AbstractImageFilter> createClone() const override;

	virtual void modify(cv::Mat& image) const override;

	//mutable bool useExistingLandmarks = false;
	//mutable std::vector<cv::Point> landmarks;
	//std::shared_ptr<FacialLandmarkDetector> landmarkDetector;
	cv::Scalar color{0,0,255,100};
	//double alpha = 0.4;
};	// LipstickColorFilter

#endif	// LIPSTICKCOLORFILTER_H