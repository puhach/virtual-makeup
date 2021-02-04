#ifndef EYECOLORFILTER_H
#define EYECOLORFILTER_H

#include "faciallandmarkfilter.h"

#include <opencv2/core.hpp>

class EyeColorFilter : public FacialLandmarkFilter
{
public:
	EyeColorFilter(std::shared_ptr<FacialLandmarkDetector> landmarkDetector) noexcept
		: FacialLandmarkFilter(std::move(landmarkDetector)) {}

	cv::Scalar getColor() const { return this->color; }

	void setColor(const cv::Scalar& color) { this->color = color; }

protected:
	// TODO: define copy/move semantics

	

private:

	virtual std::unique_ptr<AbstractImageFilter> createClone() const override;

	virtual void modify(cv::Mat& image) const override;

	void createIrisMask(const cv::Mat1b& imageGray, const std::vector<cv::Point> &eyeContour, 
		int minRadius, int maxRadius, cv::Mat1b& irisMask, cv::Point &center) const;

	void changeIrisColor(cv::Mat3b& image, const cv::Mat1b& irisMask, const cv::Point& irisCenter) const;


	void changeIrisColorPixelwise(cv::Mat3b& image, const cv::Mat1b& hueChannel, const std::vector<cv::Point>& eyeContour, 
		const cv::Point &center, int minRadius, int maxRadius) const;	// TEST!

	//int detectIris(const cv::Mat& image, cv::Point &center, int minRadius, int maxRadius) const;

	cv::Scalar color{ 55, 163, 55, 100 };
	//cv::Scalar color{ 72, 114, 75, 200 };
	//cv::Scalar color{ 100, 80, 30, 150 };
	//cv::Scalar color{ 2, 25, 46, 250 };
};	// EyeColorFilter

#endif	// EYECOLORFILTER_H