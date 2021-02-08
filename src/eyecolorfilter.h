#ifndef EYECOLORFILTER_H
#define EYECOLORFILTER_H

#include "faciallandmarkfilter.h"

#include <opencv2/core.hpp>


// A landmark-based image filter for changing color of the iris

class EyeColorFilter : public FacialLandmarkFilter
{
public:

	EyeColorFilter(std::shared_ptr<FacialLandmarkDetector> landmarkDetector, const cv::Scalar& color) //noexcept
		: FacialLandmarkFilter(std::move(landmarkDetector)) 
        , color(color) {}

	// cv::Scalar's copy constructor is not noexcept
	cv::Scalar getColor() const { return this->color; }

	void setColor(const cv::Scalar& color) { this->color = color; }

protected:
    
	EyeColorFilter() = default;
	EyeColorFilter(const EyeColorFilter&) = default;
	EyeColorFilter(EyeColorFilter&&) = default;

	EyeColorFilter& operator = (const EyeColorFilter&) = delete;
	EyeColorFilter& operator = (EyeColorFilter&&) = delete;

private:

	// Overrides the inherited method to create a correct copy of this instance
	virtual std::unique_ptr<AbstractImageFilter> createClone() const override;

	// This private function is called by multiple public overloads to apply the eye color filter in-place
	virtual void modify(cv::Mat& image) const override;

	//void createIrisMask(const cv::Mat1b& imageGray, const std::vector<cv::Point> &eyeContour, 
	//	int minRadius, int maxRadius, cv::Point &center, cv::Mat1b& irisMask) const;


	cv::Rect detectIris(const std::vector<cv::Mat1b>& channels, const std::vector<cv::Point>& eyeContour,
		int minRadius, int maxRadius, cv::Point& irisCenter, cv::Mat1b& irisMask) const;

	void changeIrisColor_Overlaying(cv::Mat3b& image, const cv::Rect& eyeRect, const cv::Mat1b& irisMask, 
		const cv::Point& irisCenter, bool blur) const;

	void changeIrisColor_Pixelwise(cv::Mat3b& image, const cv::Mat1b& hueChannel, const std::vector<cv::Point>& eyeContour,
		const cv::Point& center, int minRadius, int maxRadius) const;


	/*
	void createIrisMask(const std::vector<cv::Mat1b>& hsvChannels, const std::vector<cv::Point>& eyeContour,
		int minRadius, int maxRadius, cv::Point& center, cv::Mat1b& irisMask) const;

	void changeIrisColor_Overlaying(cv::Mat3b& image, const cv::Mat1b& irisMask, const cv::Point& irisCenter, bool blur) const;
	*/


	//cv::Scalar color{ 55, 163, 55, 100 };		// vivid green
	cv::Scalar color{ 72, 114, 75, 130 };		// pale green
	//cv::Scalar color{ 100, 80, 30, 150 };		// blue
	//cv::Scalar color{ 2, 25, 46, 250 };			// dark brown

	// do we really need these memory buffers
	//mutable cv::Mat1b eyeMask, eyeGray;
	//mutable cv::Mat3b iris;
	//mutable cv::Mat3f irisMaskF, inputF, irisF;
};	// EyeColorFilter

#endif	// EYECOLORFILTER_H
