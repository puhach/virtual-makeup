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
void LipstickColorFilter::applyInPlace(cv::Mat& image) const
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


void LipstickColorFilter::applyInPlace(cv::Mat& image) const
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
	cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{ outerPoints }, cv::Scalar(255, 255, 255));
	cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{ innerPoints }, cv::Scalar(0, 0, 0));	

	//cv::imshow("test", mask);
	//cv::waitKey();

	cv::Mat red(image.size(), CV_8UC3, cv::Scalar(0, 0, 255));
	cv::Rect r = cv::boundingRect(outerPoints);
	cv::Mat lipstick;
	cv::seamlessClone(image, red, mask, (r.tl() + r.br()) / 2, lipstick, cv::MIXED_CLONE);

	//cv::imshow("lipstick", lipstick);
	//cv::waitKey();

	
	cv::blur(mask, mask, cv::Size(5, 5));

	//cv::imshow("test", mask);
	//cv::waitKey();

	double alpha = 0.5;		// TODO: add it as a parameter
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



cv::Mat LipstickColorFilter::apply(const cv::Mat& image, const std::vector<cv::Point>& landmarks) const
{
	cv::Mat out;
	apply(image, landmarks, out);
	return out;
}

void LipstickColorFilter::apply(const cv::Mat& image, const std::vector<cv::Point>& landmarks, cv::Mat& out) const
{

	//struct ExistingLandmarksRAII
	//{
	//	ExistingLandmarksRAII(const LipstickColorFilter& filter)
	//		: filter(filter) 
	//	{
	//		this->filter.useExistingLandmarks = true;
	//	}

	//	~ExistingLandmarksRAII()
	//	{
	//		this->filter.useExistingLandmarks = false;
	//	}

	//	const LipstickColorFilter& filter;
	//} existingLandmarksRAII(*this);

	struct ExistingLandmarksRAII
	{
		ExistingLandmarksRAII(bool& useExistingLandmarks)
			: useExistingLandmarks(useExistingLandmarks)
		{
			useExistingLandmarks = true;
		}

		~ExistingLandmarksRAII()
		{
			useExistingLandmarks = false;
		}

		bool& useExistingLandmarks;
	} existingLandmarksRAII(this->useExistingLandmarks);

	this->landmarks = landmarks;

	image.copyTo(out);
	applyInPlace(out);
}