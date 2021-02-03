#include "eyecolorfilter.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>	// TEST!

#include <vector>
#include <limits>
#include <numeric>
#include <cassert>


std::unique_ptr<AbstractImageFilter> EyeColorFilter::createClone() const
{
	return std::unique_ptr<EyeColorFilter>(new EyeColorFilter(*this));
}


void EyeColorFilter::modify(cv::Mat& image) const
{
	cv::Mat3b image3b = image;	// TEST!

	CV_Assert(!image.empty());
	CV_Assert(image.type() == CV_8UC3);

	auto&& landmarks = grabLandmarks(image);
	CV_Assert(landmarks.size() == 68);

	//static constexpr std::size_t leftEyeIndices[] = { 37, 38, 40, 41 }, rightEyeIndices[] = { 43, 44, 46, 47 };
	static constexpr std::array<std::size_t, 4> leftIrisIndices{ 37, 38, 40, 41 }, rightIrisIndices{ 43, 44, 46, 47 };

	auto estimateIrisLocation = [&landmarks](const std::array<std::size_t, 4> &indices, cv::Point& center, int& minRadius, int& maxRadius)
	{
		/*center.x = center.y = 0;
		for (auto idx : indices)
		{
			center += landmarks[idx];
		}*/

		// TODO: try parallel execution
		center = std::reduce(indices.begin(), indices.end(), cv::Point(0, 0), 
			[&landmarks](const cv::Point& p, std::size_t index)
			{
				return p + landmarks[index];
			});

		assert(indices.size() > 0);
		center /= static_cast<int>(indices.size());

		std::tie(minRadius, maxRadius) = std::reduce(indices.begin(), indices.end(), std::pair<int, int>(std::numeric_limits<int>::max(), 0), 
			[&landmarks, &center](const std::pair<int, int>& p, int index) 
			{
				return std::make_pair( std::min(p.first, static_cast<int>(cv::norm(landmarks[index]-center))), 
										std::max(p.second, static_cast<int>(cv::norm(landmarks[index]-center))) );
			});
	};

	cv::Point centerLeft, centerRight;
	int minRadiusLeft, maxRadiusLeft, minRadiusRight, maxRadiusRight;
	estimateIrisLocation(leftIrisIndices, centerLeft, minRadiusLeft, maxRadiusLeft);
	estimateIrisLocation(rightIrisIndices, centerRight, minRadiusRight, maxRadiusRight);

	std::vector<cv::Point> eyePointsLeft(6), eyePointsRight(6);
	std::move(landmarks.begin() + 36, landmarks.begin() + 42, eyePointsLeft.begin());
	std::move(landmarks.begin() + 42, landmarks.begin() + 48, eyePointsRight.begin());
	/*for (std::size_t i = 0; i < 6; ++i)
	{
		eyePointsLeft[i] = std::move(landmarks[36+i]);
		eyePointsRight[i] = std::move(landmarks[42+i]);
	}*/

	//// TEST!
	//cv::Mat imagetest = image.clone();
	//cv::circle(imagetest, centerLeft, minRadiusLeft, cv::Scalar(0,255,0), 1);
	//cv::circle(imagetest, centerRight, minRadiusRight, cv::Scalar(0,255,0), 1);
	//cv::imshow("test", imagetest);
	//cv::waitKey();

	cv::Mat imageHSV;
	image.convertTo(imageHSV, image.type());
	std::vector<cv::Mat> channelsHSV;
	cv::split(image, channelsHSV);
	//cv::imshow("hue", channels[0]);
	//cv::imshow("sat", channels[1]);
	//cv::imshow("val", channels[2]);
	//cv::waitKey();

	cv::Mat1b maskLeft, maskRight;
	createIrisMask(channelsHSV[2], eyePointsLeft, minRadiusLeft, maxRadiusLeft, maskLeft, centerLeft);
	createIrisMask(channelsHSV[2], eyePointsRight, minRadiusRight, maxRadiusRight, maskRight, centerRight);

	//changeEyeColor(image, eyePointsLeft);
	//changeEyeColor(image, eyePointsRight);
	
	changeIrisColor(image3b, maskLeft, centerLeft);
	changeIrisColor(image3b, maskRight, centerRight);
}	// modify

void EyeColorFilter::createIrisMask(const cv::Mat1b& imageGray, const std::vector<cv::Point>& eyeContour, 
	int minRadius, int maxRadius, cv::Mat1b& irisMask, cv::Point& irisCenter) const
{
	CV_Assert(!eyeContour.empty());

	//cv::Mat1b mask1(imageGray.size(), 0);
	cv::Mat1b &eyeMask = irisMask;	// use the iris mask as a buffer
	eyeMask.create(imageGray.size());	// allocate new matrix if needed
	eyeMask.setTo(0);
	cv::fillConvexPoly(eyeMask, eyeContour, cv::Scalar(255));

	//cv::imshow("mask", eyeMask);
	//cv::waitKey();

	cv::dilate(eyeMask, eyeMask, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(55, 55)));

	// TODO: should it be set to white?
	cv::Mat1b eyeGray;
	imageGray.copyTo(eyeGray, eyeMask);

	//cv::imshow("eye gray", eyeGray);
	//cv::waitKey();

	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 100, 5, minRadius, maxRadius);

	// TEST!
	cv::Mat imtest;
	cv::merge(std::vector<cv::Mat>{imageGray, imageGray, imageGray}, imtest);
	for (const auto& circle : circles)
	{
		cv::circle(imtest, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
	}

	cv::imshow("circles", imtest);
	cv::waitKey();


	int radius = minRadius;
	if (!circles.empty())
	{
		int bestCircleIdx = 0;
		std::vector<double> votes(circles.size(), 0);
		//for (const auto& cl : circles)
		for (std::size_t i = 0; i < circles.size(); ++i)	// TODO: try to use reduce here too
		{
			//auto& cl = circles[i];
			cv::Point2f center(circles[i][0], circles[i][1]);
			//double voteLeft = std::exp(cv::norm(center-c1)), voteRight = std::exp(cv::norm(center-c2));
			//double& vote = cv::norm(center - c1) < cv::norm(center - c2) ? votesLeft[i] : votesRight[i];
			/*for (const auto& c : circles)
			{
				double upvote = std::exp(-cv::norm(center - cv::Point(c[0], c[1])));
				votes[i] += upvote;
			}*/

			votes[i] = std::reduce(circles.begin(), circles.end(), 0.0, [&center](double vote, const auto& circle) 
						{
							//double test = vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
							return vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
							//return test;
						});

			if (votes[i] > votes[bestCircleIdx])
				bestCircleIdx = i;
		}	// for i

		irisCenter.x = circles[bestCircleIdx][0];
		irisCenter.y = circles[bestCircleIdx][1];
		radius = circles[bestCircleIdx][2];
	}	// circles not empty

	irisMask.setTo(0);	// clear the mask
	cv::circle(irisMask, irisCenter, radius, cv::Scalar(255), -1);

	// TODO: use eye contour to cut the parts of the iris which exceed the boundaries

	// TEST!
	imtest = imageGray.clone();
	cv::circle(imtest, irisCenter, radius, cv::Scalar(0,255,0), 1);
	cv::imshow("iris", imtest);
	cv::waitKey();
	//cv::imshow("iris mask", irisMask);
	//cv::waitKey();
}	// createIrisMask

void EyeColorFilter::changeIrisColor(cv::Mat3b& image, const cv::Mat1b& irisMask, const cv::Point& irisCenter) const
{	
	cv::Mat3b iris(image.size(), cv::Vec3b( this->color[0], this->color[1], this->color[2] ));
	//cv::Mat iris(image.size(), CV_8UC3, this->color);
	cv::seamlessClone(image, iris, irisMask, irisCenter, iris, cv::MIXED_CLONE);

	//cv::imshow("test", iris);
	//cv::waitKey();



	cv::Mat3f irisMaskF, inputF, irisF;
	//mask1.convertTo(mask1F, CV_32FC3, alpha*1.0/255.0);
	//mask2.convertTo(mask2F, CV_32FC3, alpha/255.0);
	image.convertTo(inputF, CV_32FC3, 1.0 / 255);
	iris.convertTo(irisF, CV_32FC3, 1.0 / 255);

	double alpha = 0.2;		// TODO: use a class parameter
	cv::merge(std::vector<cv::Mat1b>{irisMask, irisMask, irisMask}, iris);
	iris.convertTo(irisMaskF, CV_32FC3, alpha / 255);
	//irisMask.convertTo(irisMaskF, CV_32FC3, alpha / 255);
	
	cv::erode(irisMaskF, irisMaskF, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	cv::blur(irisMaskF, irisMaskF, cv::Size(5, 5));
	

	cv::multiply(irisF, irisMaskF, irisF);
	//cv::imshow("iris", irisF);
	//cv::waitKey();


	irisMaskF.convertTo(irisMaskF, CV_32FC3, -1, +1);		// invert the mask
	cv::multiply(inputF, irisMaskF, inputF);

	inputF += irisF;
	//cv::imshow("colored", inputF);
	//cv::waitKey();

	inputF.convertTo(image, CV_8UC3, 255.0);
}	// changeIrisColor

