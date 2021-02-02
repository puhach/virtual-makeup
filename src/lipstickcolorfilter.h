#ifndef LIPSTICKCOLORFILTER_H
#define LIPSTICKCOLORFILTER_H

#include "abstractimagefilter.h"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>


class LipstickColorFilter : public AbstractImageFilter
{
public:
	LipstickColorFilter() = default;

	// TODO: define copy/move semantics

	void setLandmarks(const std::vector<cv::Point>& landmarks) { this->landmarks = landmarks; }
	void setLandmarks(std::vector<cv::Point>&& landmarks) { this->landmarks = std::move(landmarks); }

	//cv::Mat apply(const cv::Mat& image, const std::vector<cv::Point> &landmarks);

	

private:
	virtual std::unique_ptr<AbstractImageFilter> createClone() const override;

	virtual void applyInPlace(cv::Mat& image) const override;

	std::vector<cv::Point> landmarks;
};	// LipstickColorFilter

#endif	// LIPSTICKCOLORFILTER_H