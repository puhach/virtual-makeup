#ifndef LIPSTICKCOLORFILTER_H
#define LIPSTICKCOLORFILTER_H

#include "faciallandmarkfilter.h"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>


class LipstickColorFilter : public FacialLandmarkFilter
{
public:

	LipstickColorFilter(std::shared_ptr<FacialLandmarkDetector> landmarkDetector, const cv::Scalar& color) //noexcept
		: FacialLandmarkFilter(std::move(landmarkDetector))
        , color(color) {}


	// cv::Scalar's copy constructor is not noexcept
	cv::Scalar getColor() const /*noexcept*/ { return this->color; }

	void setColor(const cv::Scalar& color) { this->color = color; }

protected:

	LipstickColorFilter() = default;
	LipstickColorFilter(const LipstickColorFilter&) = default;
	LipstickColorFilter(LipstickColorFilter&&) = default;

	LipstickColorFilter& operator = (const LipstickColorFilter&) = delete;
	LipstickColorFilter& operator = (LipstickColorFilter&&) = delete;

private:

	virtual std::unique_ptr<AbstractImageFilter> createClone() const override;

	virtual void modify(cv::Mat& image) const override;

	cv::Scalar color{0,0,255,100};
};	// LipstickColorFilter

#endif	// LIPSTICKCOLORFILTER_H
