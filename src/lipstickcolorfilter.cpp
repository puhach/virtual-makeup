#include "lipstickcolorfilter.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>	// TEST!
#include <opencv2/photo.hpp>

#include <algorithm>
#include <numeric>

std::unique_ptr<AbstractImageFilter> LipstickColorFilter::createClone() const
{
	return std::unique_ptr<LipstickColorFilter>(new LipstickColorFilter(*this));
}

/*
void LipstickColorFilter::modify(cv::Mat& image) const
{
	CV_Assert(!image.empty());

	//if (this->landmarks.empty())
	CV_Assert(this->landmarks.size() == 68);

	//std::vector<int> outerIndices(12);
	//std::iota(outerIndices.begin(), );
	static constexpr int outerIndices[] = { 48,49,50,51,52,53,54,55,56,57,58,59 }, innerIndices[] = { 60,61,62,63,64,65,66,67 };

	std::vector<cv::Point> outerPoints;
	for (auto idx : outerIndices)
	{
		outerPoints.push_back(this->landmarks[idx]);
	}

	std::vector<cv::Point> innerPoints;
	for (auto idx : innerIndices)
	{
		innerPoints.push_back(this->landmarks[idx]);
	}


	cv::Mat mask(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	//cv::fillConvexPoly(mask, outerPoints, cv::Scalar(255, 255, 255));
	//cv::fillConvexPoly(mask, innerPoints, cv::Scalar(0, 0, 0));
	cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{ outerPoints }, cv::Scalar(255, 255, 255));
	cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{ innerPoints }, cv::Scalar(0, 0, 0));

	cv::imshow("test", mask);
	cv::waitKey();

	cv::blur(mask, mask, cv::Size(7, 7));

	cv::imshow("test", mask);
	cv::waitKey();

	double alpha = 0.5;		// TODO: add it as a parameter
	cv::Mat maskF;
	mask.convertTo(maskF, CV_32FC1, alpha * 1.0 / 255);


	cv::Mat inputF;
	image.convertTo(inputF, CV_32FC3, 1.0 / 255);


	//cv::Mat red(image.size(), CV_8UC3, cv::Scalar(0, 0, 255));
	//red.copyTo(out, mask);
	cv::Mat redF(image.size(), CV_32FC3, cv::Scalar(0,0,1.0));
	cv::multiply(redF, maskF, redF);
	maskF.convertTo(maskF, CV_32FC3, -1, 1);
	cv::multiply(inputF, maskF, inputF);

	inputF += redF;

	inputF.convertTo(image, CV_8UC3, 255.0);
}
*/


void LipstickColorFilter::modify(cv::Mat& image) const
{
	CV_Assert(!image.empty());

	auto&& landmarks = grabLandmarks(image);		// obtain or read existing landmarks destructively
	CV_Assert(landmarks.size() == 68);


	// Build the outer and the inner contours of the lips

	static constexpr int outerIndices[] = { 48,49,50,51,52,53,54,55,56,57,58,59 }, innerIndices[] = { 60,61,62,63,64,65,66,67 };

	std::vector<cv::Point> outerPoints;
	for (auto idx : outerIndices)
	{
		outerPoints.push_back(landmarks[idx]);
	}

	std::vector<cv::Point> innerPoints;
	for (auto idx : innerIndices)
	{
		innerPoints.push_back(landmarks[idx]);
	}


	// Create a mask for the lips region

	cv::Mat mask(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{ outerPoints }, cv::Scalar(255, 255, 255));
	cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{ innerPoints }, cv::Scalar(0, 0, 0));	

	//cv::imshow("test", mask);
	//cv::waitKey();

	// Mix the lips and the lipstick color

	//cv::Mat lipstick(image.size(), CV_8UC3, cv::Scalar(0, 0, 255));
	cv::Mat lipstick(image.size(), CV_8UC3, this->color);
	cv::Rect r = cv::boundingRect(outerPoints);
	//cv::Mat lipstick;
	cv::seamlessClone(image, lipstick, mask, (r.tl() + r.br()) / 2, lipstick, cv::MIXED_CLONE);

	//cv::imshow("lipstick", lipstick);
	//cv::waitKey();

	
	cv::blur(mask, mask, cv::Size(5, 5));

	//cv::imshow("test", mask);
	//cv::waitKey();

	//double alpha = 0.4;		// TODO: add it as a parameter
	double alpha = this->color[3] / 255.0;
	cv::Mat maskF;
	mask.convertTo(maskF, CV_32FC1, alpha * 1.0 / 255);


	cv::Mat inputF;
	image.convertTo(inputF, CV_32FC3, 1.0 / 255);

	cv::Mat lipstickF;
	lipstick.convertTo(lipstickF, CV_32FC3, 1.0/255);

	
	//cv::Mat red(image.size(), CV_8UC3, cv::Scalar(0, 0, 255));
	//red.copyTo(out, mask);
	//cv::Mat redF(image.size(), CV_32FC3, cv::Scalar(0, 0, 1.0));
	cv::multiply(lipstickF, maskF, lipstickF);

	//cv::imshow("lipstickF", lipstickF);
	//cv::waitKey();

	maskF.convertTo(maskF, CV_32FC3, -1, 1);
	cv::multiply(inputF, maskF, inputF);

	inputF += lipstickF;

	//cv::imshow("inputF", inputF);
	//cv::waitKey();

	inputF.convertTo(image, CV_8UC3, 255.0);
	//cv::imshow("test", image);
	//cv::waitKey();
}



