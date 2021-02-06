#include "eyecolorfilter.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/highgui.hpp>	// TEST!

#include <vector>
#include <limits>
#include <numeric>
#include <cassert>
#include <execution>	// execution policy


std::unique_ptr<AbstractImageFilter> EyeColorFilter::createClone() const
{
	return std::unique_ptr<EyeColorFilter>(new EyeColorFilter(*this));
}


void EyeColorFilter::modify(cv::Mat& image) const
{
	CV_Assert(!image.empty());
	CV_Assert(image.type() == CV_8UC3);
	cv::Mat3b image3b = image;	

	auto&& landmarks = grabLandmarks(image);
	CV_Assert(landmarks.size() == 68);

	//static constexpr std::size_t leftEyeIndices[] = { 37, 38, 40, 41 }, rightEyeIndices[] = { 43, 44, 46, 47 };
	static constexpr std::array<std::size_t, 4> leftIrisIndices{ 37, 38, 40, 41 }, rightIrisIndices{ 43, 44, 46, 47 };

	auto estimateIrisLocation = [&landmarks](const std::array<std::size_t, 4> &indices, cv::Point& center, int& minRadius, int& maxRadius)
	{
		center = std::accumulate(indices.begin(), indices.end(), cv::Point(0, 0), 
			[&landmarks](const cv::Point& p, std::size_t index)
			{
				return p + landmarks[index];
			});

		assert(indices.size() > 0);
		center /= static_cast<int>(indices.size());

		std::tie(minRadius, maxRadius) = std::accumulate(indices.begin(), indices.end(), std::pair<int, int>(std::numeric_limits<int>::max(), 0), 
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
	
	
	cv::Mat3b imageHSV;
	image3b.convertTo(imageHSV, image.type());
	std::vector<cv::Mat1b> channelsHSV;
	cv::split(image, channelsHSV);
	//channelsHSV.erase(channelsHSV.begin() + 1);		// TEST!
	//cv::imshow("hue", channelsHSV[0]);
	//cv::waitKey();

	//cv::imwrite("z:/value.jpg", channelsHSV[2]);

	auto changeIrisColor = [this, &channelsHSV, &image3b](const std::vector<cv::Point>& eyeContour, int minRadius, int maxRadius, cv::Point& irisCenter)
	{
		cv::Mat1b mask;
		//createIrisMask(channelsHSV[2], eyeContour, minRadius, maxRadius, irisCenter, mask);		
		createIrisMask(channelsHSV, eyeContour, minRadius, maxRadius, irisCenter, mask);
		changeIrisColor_Overlaying(image3b, mask, irisCenter, minRadius>7);
	};
	
	changeIrisColor(eyePointsLeft, minRadiusLeft, maxRadiusLeft, centerLeft);
	changeIrisColor(eyePointsRight, minRadiusRight, maxRadiusRight, centerRight);

	/*cv::Mat1b maskLeft, maskRight;
	createIrisMask(channelsHSV[0], eyePointsLeft, minRadiusLeft, maxRadiusLeft, maskLeft, centerLeft);
	createIrisMask(channelsHSV[0], eyePointsRight, minRadiusRight, maxRadiusRight, maskRight, centerRight);

	changeIrisColor(image3b, maskLeft, centerLeft);
	changeIrisColor(image3b, maskRight, centerRight);
	*/

	/*
	changeIrisColorPixelwise(image3b, channelsHSV[0], eyePointsLeft, centerLeft, minRadiusLeft, maxRadiusLeft);
	changeIrisColorPixelwise(image3b, channelsHSV[0], eyePointsRight, centerRight, minRadiusRight, maxRadiusRight);
	*/
}	// modify


void EyeColorFilter::changeIrisColor_Pixelwise(cv::Mat3b& image, const cv::Mat1b& hueChannel, const std::vector<cv::Point>& eyeContour,
	const cv::Point& center, int minRadius, int maxRadius) const
{
	cv::Mat1b eyeRegion(image.size(), 0);
	//cv::fillConvexPoly(eyeRegion, eyeContour, cv::Scalar(255));
	cv::fillConvexPoly(eyeRegion, eyeContour, 255);

	//cv::imshow("eye", eyeRegion);
	//cv::waitKey();

	cv::Mat1b irisMask(image.size(), 0);
	cv::circle(irisMask, center, maxRadius, cv::Scalar(255), -1);

	//cv::bitwise_and(irisMask, eyeRegion, irisMask);
	eyeRegion.copyTo(irisMask, irisMask);
	//cv::imshow("test", irisMask);
	//cv::waitKey();

	/*cv::Mat1b imtest;
	hueChannel.copyTo(imtest, irisMask);
	cv::imshow("part", imtest);
	cv::waitKey();*/

	cv::Mat1f hist;
	//int channels[] = { 0 }, histSize[] = { 180 };
	//cv::calcHist(&hueChannel, 1, channels, irisMask, hist, 1, )
	cv::calcHist(std::vector{ hueChannel }, { 0 }, irisMask, hist, { 180 }, {0.0f, 180.0f}, false);

	int domHue[2];
	cv::minMaxIdx(hist, nullptr, nullptr, nullptr, domHue);

	// TEST!
	cv::Mat1b devMat;
	cv::absdiff(hueChannel, cv::Scalar(domHue[0]), devMat);
	cv::Scalar meanDev = cv::mean(devMat, irisMask);

	//double sat = cv::max({ this->color[0], this->color[1], this->color[2] });
	//sat = sat > 0 ? (1 - cv::min({ this->color[0], this->color[1], this->color[2] })/sat) : 0;
	double sigmaColor = 256.0 - cv::max({ this->color[0], this->color[1], this->color[2] });

	int fromRow = std::max(0, center.y - maxRadius), toRow = std::min(image.rows-1, center.y+maxRadius);
	int fromCol = std::max(0, center.x - maxRadius), toCol = std::min(image.cols-1, center.x+maxRadius);
	for (int i = fromRow; i <= toRow; ++i)
	{
		for (int j = fromCol; j <= toCol; ++j)
		{			
			if (!irisMask.at<bool>(i, j))
				continue;


			double srcHue = hueChannel.at<uchar>(i,j);
			//double d = std::abs((srcHue - domHue[0])/meanDev[0]) + cv::norm(cv::Point(j,i) - center)/radius;
			//double d = std::abs((srcHue - domHue[0]) / sigmaColor) + cv::norm(cv::Point(j, i) - center) / minRadius;
			double devColor = std::abs(srcHue - domHue[0]), devLoc = cv::norm(cv::Point(j, i) - center);
			double weight = 1.0 * std::exp(-devColor/sigmaColor - devLoc/minRadius);
			//double weight = 1.0 * std::exp(-d);
			
			cv::Vec3b& srcColor = image.at<cv::Vec3b>(i, j);
			srcColor[0] = weight * this->color[0] + (1 - weight) * srcColor[0];
			srcColor[1] = weight * this->color[1] + (1 - weight) * srcColor[1];
			srcColor[2] = weight * this->color[2] + (1 - weight) * srcColor[2];
		}
	}

	cv::imshow("test", image);
	cv::waitKey();
}


void EyeColorFilter::createIrisMask(const std::vector<cv::Mat1b>& hsvChannels, const std::vector<cv::Point>& eyeContour,
	int minRadius, int maxRadius, cv::Point& irisCenter, cv::Mat1b& irisMask) const
{
	CV_Assert(!eyeContour.empty());
	CV_Assert(!hsvChannels.empty());

	//cv::Mat1b &eyeMask = irisMask;	// use the iris mask as a buffer	
	cv::Mat1b eyeMask(hsvChannels[0].size(), 0);
	//this->eyeMask.create(hsvChannels[0].size());	// allocate new matrix if needed
	//this->eyeMask.setTo(0);

	cv::fillConvexPoly(eyeMask, eyeContour, cv::Scalar(255));

	// TEST!
	//cv::imshow("mask", this->eyeMask);
	//cv::waitKey();

	cv::Mat1b eyeMaskDilated;
	cv::dilate(eyeMask, eyeMaskDilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*maxRadius, 2*maxRadius)));
	//cv::Mat1b eyeMaskDilated=eyeMask;	// TEST!

	cv::Mat1b eyeGray(eyeMask.size());
	std::vector<cv::Vec3f> allCircles;
	for (const auto& channel : hsvChannels)
	{
		//this->eyeGray.create(channel.size());	// allocate a new matrix if needed
		eyeGray.setTo(0);	// clear the mask
		channel.copyTo(eyeGray, eyeMaskDilated);

		// TEST!
		//cv::imshow("eye gray", eyeGray);
		//cv::waitKey();

		std::vector<cv::Vec3f> circles;
		for (int threshold = 400; threshold > 50; threshold -= 10)
		{
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 100, 5, minRadius, maxRadius);
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 100, 5, minRadius, minRadius + 0.7*(maxRadius-minRadius));	// TEST!
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 300, 5, minRadius, maxRadius);
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 240, 5, minRadius, maxRadius);
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 270, 5, minRadius, maxRadius);
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 293, 5, minRadius, maxRadius);
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 292, 5, minRadius, maxRadius);	// !
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 300, 4, minRadius, maxRadius);

			cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, threshold, 5, minRadius, maxRadius);
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, threshold, 3, minRadius, maxRadius);	// TEST!

			// Filter the circles: the iris center must lie within the eye contour
			auto last = std::remove_if(std::execution::par_unseq, circles.begin(), circles.end(), 
						[&eyeContour](const auto& circle)
						{
							return cv::pointPolygonTest(eyeContour, cv::Point2f(circle[0], circle[1]), false) < 0;
						});
			//allCircles.erase(last, allCircles.end());
			if (last != circles.begin())	// not empty vector
			{
				/*
				// TEST!
				cv::Mat imcircles;
				cv::merge(std::vector<cv::Mat>{channel, channel, channel}, imcircles);
				//cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
				for (const auto& circle : circles)
				{
					cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
					//cv::circle(imcircles, cv::Point(circle[0], circle[1]), 1, cv::Scalar(255,0,0), -1);
				}
				///cv::imwrite("z:/circles.jpg", imcircles);
				cv::imshow("circles " + std::to_string(t++), imcircles);
				cv::waitKey();
				*/
				std::move(circles.begin(), last, std::back_inserter(allCircles));
				break;		
			}	// not empty 					

		}	// for threshold
		
	}	// for channels

	//// Filter the circles: the iris center must lie within the eye contour
	//auto last = std::remove_if(allCircles.begin(), allCircles.end(), [&eyeContour](const auto& circle)
	//	{
	//		return cv::pointPolygonTest(eyeContour, cv::Point2f(circle[0], circle[1]), false) < 0;
	//	});
	//allCircles.erase(last, allCircles.end());

	//// TEST!
	//cv::Mat imcircles;
	//cv::merge(std::vector<cv::Mat>{hsvChannels[0], hsvChannels[0], hsvChannels[0]}, imcircles);
	////cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
	//for (const auto& circle : allCircles)
	//{
	//	cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
	//	//cv::circle(imcircles, cv::Point(circle[0], circle[1]), 1, cv::Scalar(255,0,0), -1);
	//}
	////cv::imwrite("z:/circles.jpg", imcircles);
	//cv::imshow("circles filtered", imcircles);
	//cv::waitKey();

	int radius = minRadius;
	if (!allCircles.empty())
	{
		//int bestCircleIdx = 0;
		std::vector<double> ranks(allCircles.size(), 0);
				
		std::transform(std::execution::par_unseq, allCircles.begin(), allCircles.end(), ranks.begin(), 
			[&allCircles](const cv::Vec3f& circleI) 
			{
				return std::accumulate(allCircles.begin(), allCircles.end(), 0.0,
							[&center = cv::Point2f(circleI[0], circleI[1])](double rank, const cv::Vec3f& circleJ)
							{
								return rank + std::exp(-cv::norm(center - cv::Point2f(circleJ[0], circleJ[1])));
							});	// accumulate
			});	// transform

		auto bestCircleIdx = std::max_element(std::execution::par, ranks.begin(), ranks.end()) - ranks.begin();
		assert(bestCircleIdx >= 0);

		//std::transform_reduce(allCircles.begin(), allCircles.end(), std::pair{0, 0.0}, 
		//	[](const auto& p, double rank)	// reduce
		//	{
		//		if (rank < )
		//	},
		//	[&allCircles](const auto& circle)	// transform
		//	{				
		//		return std::reduce(allCircles.begin(), allCircles.end(), 0.0,
		//			[&center = circle](double vote, const auto& circle)
		//			{
		//				//double test = vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
		//				return vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
		//				//return test;
		//			});
		//	});	// transform_reduce

		/*
		for (std::size_t i = 0; i < allCircles.size(); ++i)	// TODO: try to use reduce here too
		{
			//cv::Point2f center(allCircles[i][0], allCircles[i][1]);


			// The iris center must lie within the eye contour
			//assert(cv::pointPolygonTest(eyeContour, center, false) >= 0);

			ranks[i] = std::reduce(allCircles.begin(), allCircles.end(), 0.0, 
				[&center=cv::Point2f(allCircles[i][0], allCircles[i][1])](double vote, const auto& circle)
				{
					//double test = vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
					return vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
					//return test;
				});

			if (ranks[i] > ranks[bestCircleIdx])
				bestCircleIdx = i;
		}	// for i
		*/

		// In case all the circles have very low ranks, it is better to go with our initial estimate 
		assert(ranks[bestCircleIdx] > 0);
		irisCenter.x = allCircles[bestCircleIdx][0];
		irisCenter.y = allCircles[bestCircleIdx][1];
		radius = allCircles[bestCircleIdx][2];
	}	// circles not empty

	// Fill the iris circle
	//CV_Assert(!irisMask.empty());	// check that we have created the iris mask
	irisMask.create(hsvChannels[0].size());	// make sure the matrix is allocated
	irisMask.setTo(0);	// clear the mask
	cv::circle(irisMask, irisCenter, radius, cv::Scalar(255), -1);

	//// Fill the eye region	
	//// TODO: avoid redrawing
	//eyeGray.setTo(0);
	//cv::fillConvexPoly(eyeGray, eyeContour, cv::Scalar(255));

	// Use the eye contour to cut the parts of the iris which exceed the boundaries
	cv::bitwise_and(irisMask, eyeMask, irisMask);

	//// TEST!
	//cv::Mat imtest = hsvChannels[0].clone();
	//cv::circle(imtest, irisCenter, radius, cv::Scalar(0, 255, 0), 1);
	//cv::imshow("iris", imtest);
	//cv::waitKey();
	//cv::imshow("iris mask", irisMask);
	//cv::waitKey();
}	// createIrisMask

/*
void EyeColorFilter::createIrisMask(const cv::Mat1b& imageGray, const std::vector<cv::Point>& eyeContour, 
	int minRadius, int maxRadius, cv::Point& irisCenter, cv::Mat1b& irisMask) const
{
	CV_Assert(!eyeContour.empty());

	//cv::Mat1b mask1(imageGray.size(), 0);
	//cv::Mat1b &eyeMask = irisMask;	// use the iris mask as a buffer	
	this->eyeMask.create(imageGray.size());	// allocate new matrix if needed
	this->eyeMask.setTo(0);

	cv::fillConvexPoly(eyeMask, eyeContour, cv::Scalar(255));

	// TEST!
	//cv::imshow("mask", this->eyeMask);
	//cv::waitKey();

	cv::Mat1b &eyeMaskDilated = irisMask;	// use the iris mask as a buffer
	//cv::dilate(this->eyeMask, eyeMask, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(55, 55)));
	//cv::dilate(this->eyeMask, eyeMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(55, 55)));
	cv::dilate(this->eyeMask, eyeMaskDilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*maxRadius, 2*maxRadius)));

	// TODO: should it be set to white?
	//cv::Mat1b eyeGray;
	cv::Mat1b& eyeGray = irisMask;	// reuse the iris mask memory again (it seems to be ok if the mask and the destination overlap)
	//imageGray.copyTo(this->eyeGray, eyeMaskDilated);
	imageGray.copyTo(eyeGray, eyeMaskDilated);

	// TEST!
	//cv::imshow("eye gray", eyeGray);
	//cv::waitKey();

	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 100, 5, minRadius, maxRadius);
	//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 100, 5, minRadius, minRadius + 0.7*(maxRadius-minRadius));	// TEST!
	//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, 100, 3, minRadius, maxRadius);

	// TEST!
	cv::Mat imcircles;
	cv::merge(std::vector<cv::Mat>{imageGray, imageGray, imageGray}, imcircles);
	//cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
	for (const auto& circle : circles)
	{
		cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
		//cv::circle(imcircles, cv::Point(circle[0], circle[1]), 1, cv::Scalar(255,0,0), -1);
	}
	///cv::imwrite("z:/circles.jpg", imcircles);
	cv::imshow("circles", imcircles);
	cv::waitKey();


	// Filter the circles: the iris center must lie within the eye contour
	auto last = std::remove_if(circles.begin(), circles.end(), [&eyeContour](const auto& circle) 
		{
			return cv::pointPolygonTest(eyeContour, cv::Point2f(circle[0], circle[1]), false) <= 0;
		});
	circles.erase(last, circles.end());

	// TEST!
	cv::merge(std::vector<cv::Mat>{imageGray, imageGray, imageGray}, imcircles);
	//cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
	for (const auto& circle : circles)
	{
		cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
		//cv::circle(imcircles, cv::Point(circle[0], circle[1]), 1, cv::Scalar(255,0,0), -1);
	}
	///cv::imwrite("z:/circles.jpg", imcircles);
	cv::imshow("circles filtered", imcircles);
	cv::waitKey();

	int radius = minRadius;
	if (!circles.empty())
	{
		int bestCircleIdx = 0;
		std::vector<double> ranks(circles.size(), 0);

		for (std::size_t i = 0; i < circles.size(); ++i)	// TODO: try to use reduce here too
		{
			//auto& cl = circles[i];
			cv::Point2f center(circles[i][0], circles[i][1]);
			
			//if (cv::norm(center - cv::Point2f(irisCenter)) > minRadius)
			//	continue;

			// The iris center must lie within the eye contour
			assert(cv::pointPolygonTest(eyeContour, center, false) > 0);
			//if (cv::pointPolygonTest(eyeContour, center, false) <= 0)	
			//	continue;

			ranks[i] = std::reduce(circles.begin(), circles.end(), 0.0, [&center](double vote, const auto& circle) 
						{
							//double test = vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
							return vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
							//return test;
						});

			if (ranks[i] > ranks[bestCircleIdx])
				bestCircleIdx = i;
		}	// for i

		// In case all the circles have very low ranks, it is better to go with our initial estimate 
		if (ranks[bestCircleIdx] > 0)
		{
			irisCenter.x = circles[bestCircleIdx][0];
			irisCenter.y = circles[bestCircleIdx][1];
			radius = circles[bestCircleIdx][2];
		}
	}	// circles not empty

	// Fill the iris circle
	CV_Assert(!irisMask.empty());	// check that we have created the iris mask
	irisMask.setTo(0);	// clear the mask
	cv::circle(irisMask, irisCenter, radius, cv::Scalar(255), -1);

	//// Fill the eye region	
	//// TODO: avoid redrawing
	//eyeGray.setTo(0);
	//cv::fillConvexPoly(eyeGray, eyeContour, cv::Scalar(255));

	// Use the eye contour to cut the parts of the iris which exceed the boundaries
	cv::bitwise_and(irisMask, this->eyeMask, irisMask);

	// TEST!
	cv::Mat imtest = imageGray.clone();
	cv::circle(imtest, irisCenter, radius, cv::Scalar(0,255,0), 1);
	cv::imshow("iris", imtest);
	cv::waitKey();
	//cv::imshow("iris mask", irisMask);
	//cv::waitKey();
}	// createIrisMask
*/

void EyeColorFilter::changeIrisColor_Overlaying(cv::Mat3b& image, const cv::Mat1b& irisMask, const cv::Point& irisCenter, bool blur) const
{	
	cv::Mat3b iris(image.size(), cv::Vec3b( this->color[0], this->color[1], this->color[2] ));
	//this->iris.create(image.size());
	//this->iris.setTo(cv::Vec3b(this->color[0], this->color[1], this->color[2]));
	//cv::Mat iris(image.size(), CV_8UC3, this->color);
	cv::seamlessClone(image, iris, irisMask, irisCenter, iris, cv::MIXED_CLONE);

	//cv::imshow("test", this->iris);
	//cv::waitKey();



	cv::Mat3f irisMaskF, inputF, irisF;
	//mask1.convertTo(mask1F, CV_32FC3, alpha*1.0/255.0);
	//mask2.convertTo(mask2F, CV_32FC3, alpha/255.0);
	image.convertTo(inputF, CV_32FC3, 1.0 / 255);
	iris.convertTo(irisF, CV_32FC3, 1.0 / 255);

	double alpha = this->color[3] / 255;	// opacity 0..1
	cv::merge(std::vector<cv::Mat1b>{irisMask, irisMask, irisMask}, iris);
	iris.convertTo(irisMaskF, CV_32FC3, alpha * 1.0 / 255);
	//irisMask.convertTo(irisMaskF, CV_32FC3, alpha / 255);
	
	// Blur the iris mask unless it's too small
	if (blur)
	{
		cv::erode(irisMaskF, irisMaskF, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
		//cv::blur(irisMaskF, irisMaskF, cv::Size(7, 7));
		cv::GaussianBlur(irisMaskF, irisMaskF, cv::Size(7, 7), 0, 0);
	}

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

