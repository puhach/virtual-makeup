#include "lipstickcolorfilter.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <algorithm>
#include <numeric>

std::unique_ptr<AbstractImageFilter> LipstickColorFilter::createClone() const
{
	return std::unique_ptr<LipstickColorFilter>(new LipstickColorFilter(*this));
}

// Applies the lipstick color filter to the input image in-place
void LipstickColorFilter::modify(cv::Mat& image) const
{
	CV_Assert(!image.empty());
	CV_Assert(image.type() == CV_8UC3);

	auto&& landmarks = grabLandmarks(image);		// obtain or read existing landmarks destructively
	CV_Assert(landmarks.size() == 68);


	// Build the outer and the inner contours of the lips

	std::vector<cv::Point> innerPoints, outerPoints;
	std::move(landmarks.begin()+48, landmarks.begin()+59+1, std::back_inserter(outerPoints));
	std::move(landmarks.begin()+60, landmarks.begin()+67+1, std::back_inserter(innerPoints));

	// Create a mask for the lips region

	cv::Mat mask(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{ outerPoints }, cv::Scalar(255, 255, 255));
	cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{ innerPoints }, cv::Scalar(0, 0, 0));	

	// Mix the lips and the lipstick color to create a new texture

	cv::Mat lipstick(image.size(), CV_8UC3, this->color);
	cv::Rect r = cv::boundingRect(outerPoints);
	cv::seamlessClone(image, lipstick, mask, (r.tl() + r.br()) / 2, lipstick, cv::MIXED_CLONE);

	// Make edges softer
	cv::blur(mask, mask, cv::Size(5, 5));

	// Perform alpha matting (this way we change the color of the lips but preserve original details)

	double alpha = this->color[3] / 255.0;
	cv::Mat maskF;
	mask.convertTo(maskF, CV_32FC1, alpha * 1.0 / 255);

	cv::Mat inputF;
	image.convertTo(inputF, CV_32FC3, 1.0 / 255);

	cv::Mat lipstickF;
	lipstick.convertTo(lipstickF, CV_32FC3, 1.0/255);

	cv::multiply(lipstickF, maskF, lipstickF);

	maskF.convertTo(maskF, CV_32FC3, -1, 1);	// invert the mask
	cv::multiply(inputF, maskF, inputF);

	inputF += lipstickF;

	inputF.convertTo(image, CV_8UC3, 255.0);
}	// modify



