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


// This private function is called by multiple public overloads to apply the eye color filter in-place.
// It is virtual and can be overridden in descendants to alter filter behavior.
void EyeColorFilter::modify(cv::Mat& image) const
{
	CV_Assert(!image.empty());
	CV_Assert(image.type() == CV_8UC3);
	cv::Mat3b image3b = image;	

	// Detect or destructively read existing facial landmarks
	auto&& landmarks = grabLandmarks(image);
	CV_Assert(landmarks.size() == 68);


	// Estimate the iris centers and radii using the landmarks

	static constexpr std::array<std::size_t, 4> leftIrisIndices{ 37, 38, 40, 41 }, rightIrisIndices{ 43, 44, 46, 47 };

	auto estimateIrisLocation = [&landmarks](const std::array<std::size_t, 4> &indices, cv::Point& center, int& minRadius, int& maxRadius)
	{
		// The center of the iris is expected to be located in the middle of inner landmarks on the eye contour

		center = std::accumulate(indices.begin(), indices.end(), cv::Point(0, 0), 
			[&landmarks](const cv::Point& p, std::size_t index)
			{
				return p + landmarks[index];
			});

		assert(indices.size() > 0);
		center /= static_cast<int>(indices.size());

		// The radius of the iris is expected to be between the closest and the most distant inner landmark with respect to the estimated center
		std::tie(minRadius, maxRadius) = std::accumulate(indices.begin(), indices.end(), std::pair<int, int>(std::numeric_limits<int>::max(), 0), 
			[&landmarks, &center](const std::pair<int, int>& p, std::size_t index) 
			{
				return std::make_pair( std::min(p.first, static_cast<int>(cv::norm(landmarks[index]-center))), 
										std::max(p.second, static_cast<int>(cv::norm(landmarks[index]-center))) );
			});
	};

	cv::Point centerLeft, centerRight;
	int minRadiusLeft, maxRadiusLeft, minRadiusRight, maxRadiusRight;
	estimateIrisLocation(leftIrisIndices, centerLeft, minRadiusLeft, maxRadiusLeft);
	estimateIrisLocation(rightIrisIndices, centerRight, minRadiusRight, maxRadiusRight);

	// Build contours for the left and right eye
	std::vector<cv::Point> eyePointsLeft(6), eyePointsRight(6);
	std::move(landmarks.begin() + 36, landmarks.begin() + 41 + 1, eyePointsLeft.begin());
	std::move(landmarks.begin() + 42, landmarks.begin() + 47 + 1, eyePointsRight.begin());
	
	// Split the image into hue, saturation, and value channels to be processed separately 
	cv::Mat3b imageHSV;
	image3b.convertTo(imageHSV, image.type());
	std::vector<cv::Mat1b> hsvChannels;
	cv::split(image, hsvChannels);
	

	// Use the eye contour along with the approximated iris parameters to find the iris more precisely. Once we know its position,
	// we can change the color to the one specified by the user.

	auto changeIrisColor = [this, &hsvChannels, &image3b](const std::vector<cv::Point>& eyeContour, int minRadius, int maxRadius, cv::Point& irisCenter)
	{
		cv::Mat1b mask;
		cv::Rect eyeRect = detectIris(hsvChannels, eyeContour, minRadius, maxRadius, irisCenter, mask);
		changeIrisColor_Overlaying(image3b, eyeRect, mask, irisCenter, minRadius > 7);

		/*
		// Another working method
		changeIrisColor_Pixelwise(image3b, hsvChannels[0], eyeContour, irisCenter, minRadius, maxRadius);
		*/
	};

	changeIrisColor(eyePointsLeft, minRadiusLeft, maxRadiusLeft, centerLeft);
	changeIrisColor(eyePointsRight, minRadiusRight, maxRadiusRight, centerRight);
}	// modify


cv::Rect EyeColorFilter::detectIris(const std::vector<cv::Mat1b>& channels, const std::vector<cv::Point>& eyeContour,
	int minRadius, int maxRadius, cv::Point& irisCenter, cv::Mat1b& irisMask) const
{
	CV_Assert(!eyeContour.empty());
	CV_Assert(!channels.empty());

	// Use a slightly padded bounding rectange of the eye to restrict iris detection only in that region

	cv::Rect eyeRect = cv::boundingRect(eyeContour);
	eyeRect.x = std::max(0, eyeRect.x - maxRadius);
	eyeRect.y = std::max(0, eyeRect.y - maxRadius);
	eyeRect.width = std::min(channels[0].cols - eyeRect.x, eyeRect.width + 2 * maxRadius);
	eyeRect.height = std::min(channels[0].rows - eyeRect.y, eyeRect.height + 2 * maxRadius);

	std::vector<cv::Vec3f> allCircles;
	std::vector<cv::Vec3f> circles;
	for (const auto& channel : channels)
	{
		cv::Mat1b eyeGray = channel(eyeRect);

		//// TEST!
		//cv::imshow("eye gray", eyeGray);
		//cv::waitKey();

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

			circles.clear();
			//std::vector<cv::Vec3f> circles;	// TEST!
			// TODO: try HOUGH_GRADIENT_ALT (requires opencv 4.4)
			cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, threshold, 5, minRadius, maxRadius);
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1.5, 1, threshold, 5, minRadius, maxRadius);

			// Filter the circles: the iris center must lie within the eye contour
			auto last = std::remove_if(std::execution::par_unseq, circles.begin(), circles.end(),
				[&eyeContour, &eyeRect](const auto& circle)
				{
					return cv::pointPolygonTest(eyeContour, cv::Point2f(eyeRect.x + circle[0], eyeRect.y + circle[1]), false) < 0;
				});
			//allCircles.erase(last, allCircles.end());
			if (last != circles.begin())	// not empty vector
			{

				//// TEST!
				//cv::Mat imcircles;
				//cv::merge(std::vector<cv::Mat>{channel, channel, channel}, imcircles);
				////cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
				//for (const auto& circle : circles)
				//{
				//	cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
				//	//cv::circle(imcircles, cv::Point(circle[0], circle[1]), 1, cv::Scalar(255,0,0), -1);
				//}
				/////cv::imwrite("z:/circles.jpg", imcircles);
				//cv::imshow("circles " + std::to_string(t++), imcircles);
				//cv::waitKey();

				std::move(circles.begin(), last, std::back_inserter(allCircles));
				break;
			}	// not empty 					

		}	// for threshold

	}	// for channels


	//// TEST!
	//cv::Mat imcircles;
	//cv::merge(std::vector<cv::Mat>{hsvChannels[0], hsvChannels[0], hsvChannels[0]}, imcircles);
	////cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
	//for (const auto& circle : allCircles)
	//{
	//	cv::circle(imcircles, cv::Point(eyeRect.x+circle[0], eyeRect.y+circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);	// TEST!
	//	//cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
	//	//cv::circle(imcircles, cv::Point(circle[0], circle[1]), 1, cv::Scalar(255,0,0), -1);
	//}
	////cv::imwrite("z:/circles.jpg", imcircles);
	//cv::imshow("circles filtered", imcircles);
	//cv::waitKey();

	// Select the best iris circle

	//cv::Point irisCenter{ eyeCenter.x - eyeRect.x, eyeCenter.y - eyeRect.y };
	irisCenter.x -= eyeRect.x;
	irisCenter.y -= eyeRect.y;
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

		auto bestCircleIdx = std::max_element(std::execution::par_unseq, ranks.begin(), ranks.end()) - ranks.begin();
		assert(bestCircleIdx >= 0);

		//for (std::size_t i = 0; i < allCircles.size(); ++i)	// TODO: try to use reduce here too
		//{
		//	//cv::Point2f center(allCircles[i][0], allCircles[i][1]);


		//	// The iris center must lie within the eye contour
		//	//assert(cv::pointPolygonTest(eyeContour, center, false) >= 0);

		//	ranks[i] = std::reduce(allCircles.begin(), allCircles.end(), 0.0, 
		//		[&center=cv::Point2f(allCircles[i][0], allCircles[i][1])](double vote, const auto& circle)
		//		{
		//			//double test = vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
		//			return vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
		//			//return test;
		//		});

		//	if (ranks[i] > ranks[bestCircleIdx])
		//		bestCircleIdx = i;
		//}	// for i

		// In case all the circles have very low ranks, it is better to go with our initial estimate 
		assert(ranks[bestCircleIdx] > 0);
		//irisCenter.x = static_cast<int>(eyeRect.x + allCircles[bestCircleIdx][0]);
		irisCenter.x = static_cast<int>(allCircles[bestCircleIdx][0]);
		//irisCenter.y = static_cast<int>(eyeRect.y + allCircles[bestCircleIdx][1]);
		irisCenter.y = static_cast<int>(allCircles[bestCircleIdx][1]);
		radius = static_cast<int>(allCircles[bestCircleIdx][2]);
	}	// circles not empty

	// Fill the iris circle
	//CV_Assert(!irisMask.empty());	// check that we have created the iris mask
	//irisMask.create(channels[0].size());	// make sure the matrix is allocated
	irisMask.create(eyeRect.size());	// make sure the matrix is allocated
	irisMask.setTo(0);	// clear the mask
	//cv::Mat1b irisMask(eyeRect.size(), 0);
	cv::circle(irisMask, irisCenter, radius, cv::Scalar(255), -1);

	// Use the eye contour to clip the parts of the iris exceeding eye boundaries
	cv::Mat1b eyeMask(channels[0].size(), 0);
	cv::fillConvexPoly(eyeMask, eyeContour, cv::Scalar(255));
	cv::bitwise_and(irisMask, eyeMask(eyeRect), irisMask);

	//// TEST!
	//cv::Mat imtest = hsvChannels[0].clone();
	//cv::circle(imtest, irisCenter, radius, cv::Scalar(0, 255, 0), 1);
	//cv::imshow("iris", imtest);
	//cv::waitKey();
	//cv::imshow("iris mask", irisMask);
	//cv::waitKey();

	return eyeRect;
}	// detectIris

void EyeColorFilter::changeIrisColor_Overlaying(cv::Mat3b& image, const cv::Rect& eyeRect, const cv::Mat1b& irisMask,
	const cv::Point& irisCenter, bool blur) const
{
	CV_Assert(!image.empty());
	CV_Assert(!eyeRect.empty());
	CV_Assert(!irisMask.empty());

	// Create a new iris texture by mixing the original iris texture and the new color
	cv::Mat3b irisTexture(eyeRect.size(), cv::Vec3b(static_cast<uchar>(this->color[0]), static_cast<uchar>(this->color[1]), static_cast<uchar>(this->color[2])));
	cv::Mat3b imageROI = image(eyeRect);
	cv::seamlessClone(imageROI, irisTexture, irisMask, irisCenter, irisTexture, cv::MIXED_CLONE);

	//cv::imshow("test", this->iris);
	//cv::waitKey();


	// Prepare the input image ROI, the iris texture, and the iris mask for alpha matting

	cv::Mat3f irisMaskF, inputF, irisF;
	imageROI.convertTo(inputF, CV_32FC3, 1.0 / 255);
	irisTexture.convertTo(irisF, CV_32FC3, 1.0 / 255);

	double alpha = this->color[3] / 255;	// opacity 0..1
	cv::merge(std::vector<cv::Mat1b>{irisMask, irisMask, irisMask}, irisTexture);
	irisTexture.convertTo(irisMaskF, CV_32FC3, alpha * 1.0 / 255);

	// Blur the iris mask unless it's too small
	if (blur)
	{
		cv::erode(irisMaskF, irisMaskF, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
		//cv::blur(irisMaskF, irisMaskF, cv::Size(7, 7));
		cv::GaussianBlur(irisMaskF, irisMaskF, cv::Size(7, 7), 0, 0);
	}

	// Perform alpha matting

	cv::multiply(irisF, irisMaskF, irisF);
	//cv::imshow("iris", irisF);
	//cv::waitKey();

	irisMaskF.convertTo(irisMaskF, CV_32FC3, -1, +1);		// invert the mask
	cv::multiply(inputF, irisMaskF, inputF);

	inputF += irisF;
	//cv::imshow("colored", inputF);
	//cv::waitKey();

	inputF.convertTo(imageROI, CV_8UC3, 255.0);
}	// changeIrisColor_Overlaying


void EyeColorFilter::changeIrisColor_Pixelwise(cv::Mat3b& image, const cv::Mat1b& hueChannel, const std::vector<cv::Point>& eyeContour,
	const cv::Point& center, int minRadius, int maxRadius) const
{
	CV_Assert(!image.empty());
	CV_Assert(!hueChannel.empty());

	cv::Mat1b eyeRegion(image.size(), 0);
	cv::fillConvexPoly(eyeRegion, eyeContour, 255);

	cv::Mat1b irisMask(image.size(), 0);
	cv::circle(irisMask, center, maxRadius, cv::Scalar(255), -1);
	eyeRegion.copyTo(irisMask, irisMask);
	//cv::imshow("test", irisMask);
	//cv::waitKey();

	// Build the histogram of hues 
	cv::Mat1f hist;
	cv::calcHist(std::vector{ hueChannel }, { 0 }, irisMask, hist, { 180 }, { 0.0f, 180.0f }, false);

	// Find the dominant hue
	int domHue[2];
	cv::minMaxIdx(hist, nullptr, nullptr, nullptr, domHue);

	//// TEST!
	//cv::Mat1b devMat;
	//cv::absdiff(hueChannel, cv::Scalar(domHue[0]), devMat);
	//cv::Scalar meanDev = cv::mean(devMat, irisMask);

	// The color normalization constant is higher when the new color is less vivid 
	// (this way we give a new color more weight in the final color mix)
	double sigmaColor = 256.0 - cv::max({ this->color[0], this->color[1], this->color[2] });
	//double sat = cv::max({ this->color[0], this->color[1], this->color[2] });
	//sat = sat > 0 ? (1 - cv::min({ this->color[0], this->color[1], this->color[2] })/sat) : 0;

	// Process the iris region pixel by pixel
	int fromRow = std::max(0, center.y - maxRadius), toRow = std::min(image.rows - 1, center.y + maxRadius);
	int fromCol = std::max(0, center.x - maxRadius), toCol = std::min(image.cols - 1, center.x + maxRadius);
	for (int i = fromRow; i <= toRow; ++i)
	{
		for (int j = fromCol; j <= toCol; ++j)
		{
			if (!irisMask.at<bool>(i, j))	// skip pixels outside the iris mask
				continue;

			// Calculate the new color weight: it is higher if the current pixel color is close to the dominant color
			// and if the current pixel location is close to the estimated iris center
			double srcHue = hueChannel.at<uchar>(i, j);
			//double d = std::abs((srcHue - domHue[0])/meanDev[0]) + cv::norm(cv::Point(j,i) - center)/radius;
			//double d = std::abs((srcHue - domHue[0]) / sigmaColor) + cv::norm(cv::Point(j, i) - center) / minRadius;
			double devColor = std::abs(srcHue - domHue[0]), devLoc = cv::norm(cv::Point(j, i) - center);
			double weight = 1.0 * std::exp(-devColor / sigmaColor - devLoc / minRadius);
			//double weight = this->color[3]/255 * std::exp(-devColor / sigmaColor - devLoc / minRadius);

			// Mix the original color of the pixel and the new color with respect to the calculated weight
			cv::Vec3b& srcColor = image.at<cv::Vec3b>(i, j);
			assert(weight * this->color[0] + (1 - weight) * srcColor[0] <= 255);
			assert(weight * this->color[1] + (1 - weight) * srcColor[0] <= 255);
			assert(weight * this->color[2] + (1 - weight) * srcColor[0] <= 255);
			srcColor[0] = static_cast<uchar>(weight * this->color[0] + (1 - weight) * srcColor[0]);
			srcColor[1] = static_cast<uchar>(weight * this->color[1] + (1 - weight) * srcColor[1]);
			srcColor[2] = static_cast<uchar>(weight * this->color[2] + (1 - weight) * srcColor[2]);
		}	// for j
	}	// for i

	/*cv::imshow("test", image);
	cv::waitKey();*/
}	// changeIrisColor_Pixelwise


/*
void EyeColorFilter::changeIrisColor_Overlaying(cv::Mat3b& image, const std::vector<cv::Mat1b>& hsvChannels, 
	const std::vector<cv::Point>& eyeContour, const cv::Point& eyeCenter, int minRadius, int maxRadius, bool blur) const
{
	CV_Assert(!eyeContour.empty());
	CV_Assert(!hsvChannels.empty());

	// Use a slightly padded bounding rectange of the eye to restrict iris detection only in that region

	cv::Rect eyeRect = cv::boundingRect(eyeContour);
	eyeRect.x = std::max(0, eyeRect.x - maxRadius);
	eyeRect.y = std::max(0, eyeRect.y - maxRadius);
	eyeRect.width = std::min(image.cols - eyeRect.x, eyeRect.width + 2 * maxRadius);
	eyeRect.height = std::min(image.rows - eyeRect.y, eyeRect.height + 2 * maxRadius);

	std::vector<cv::Vec3f> allCircles;
	std::vector<cv::Vec3f> circles;
	for (const auto& channel : hsvChannels)
	{
		cv::Mat1b eyeGray = channel(eyeRect);

		//// TEST!
		//cv::imshow("eye gray", eyeGray);
		//cv::waitKey();

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
			
			circles.clear();
			//std::vector<cv::Vec3f> circles;	// TEST!
			// TODO: try HOUGH_GRADIENT_ALT (requires opencv 4.4)
			cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, threshold, 5, minRadius, maxRadius);
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1.5, 1, threshold, 5, minRadius, maxRadius);

			// Filter the circles: the iris center must lie within the eye contour
			auto last = std::remove_if(std::execution::par_unseq, circles.begin(), circles.end(),
				[&eyeContour, &eyeRect](const auto& circle)
				{
					return cv::pointPolygonTest(eyeContour, cv::Point2f(eyeRect.x + circle[0], eyeRect.y + circle[1]), false) < 0;
				});
			//allCircles.erase(last, allCircles.end());
			if (last != circles.begin())	// not empty vector
			{

				//// TEST!
				//cv::Mat imcircles;
				//cv::merge(std::vector<cv::Mat>{channel, channel, channel}, imcircles);
				////cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
				//for (const auto& circle : circles)
				//{
				//	cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
				//	//cv::circle(imcircles, cv::Point(circle[0], circle[1]), 1, cv::Scalar(255,0,0), -1);
				//}
				/////cv::imwrite("z:/circles.jpg", imcircles);
				//cv::imshow("circles " + std::to_string(t++), imcircles);
				//cv::waitKey();

				std::move(circles.begin(), last, std::back_inserter(allCircles));
				break;
			}	// not empty 					

		}	// for threshold

	}	// for channels


	//// TEST!
	//cv::Mat imcircles;
	//cv::merge(std::vector<cv::Mat>{hsvChannels[0], hsvChannels[0], hsvChannels[0]}, imcircles);
	////cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
	//for (const auto& circle : allCircles)
	//{
	//	cv::circle(imcircles, cv::Point(eyeRect.x+circle[0], eyeRect.y+circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);	// TEST!
	//	//cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
	//	//cv::circle(imcircles, cv::Point(circle[0], circle[1]), 1, cv::Scalar(255,0,0), -1);
	//}
	////cv::imwrite("z:/circles.jpg", imcircles);
	//cv::imshow("circles filtered", imcircles);
	//cv::waitKey();

	// Select the best iris circle

	cv::Point irisCenter{ eyeCenter.x-eyeRect.x, eyeCenter.y-eyeRect.y };
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

		auto bestCircleIdx = std::max_element(std::execution::par_unseq, ranks.begin(), ranks.end()) - ranks.begin();
		assert(bestCircleIdx >= 0);

		//for (std::size_t i = 0; i < allCircles.size(); ++i)	// TODO: try to use reduce here too
		//{
		//	//cv::Point2f center(allCircles[i][0], allCircles[i][1]);


		//	// The iris center must lie within the eye contour
		//	//assert(cv::pointPolygonTest(eyeContour, center, false) >= 0);

		//	ranks[i] = std::reduce(allCircles.begin(), allCircles.end(), 0.0, 
		//		[&center=cv::Point2f(allCircles[i][0], allCircles[i][1])](double vote, const auto& circle)
		//		{
		//			//double test = vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
		//			return vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
		//			//return test;
		//		});

		//	if (ranks[i] > ranks[bestCircleIdx])
		//		bestCircleIdx = i;
		//}	// for i

		// In case all the circles have very low ranks, it is better to go with our initial estimate 
		assert(ranks[bestCircleIdx] > 0);
		//irisCenter.x = static_cast<int>(eyeRect.x + allCircles[bestCircleIdx][0]);
		irisCenter.x = static_cast<int>(allCircles[bestCircleIdx][0]);
		//irisCenter.y = static_cast<int>(eyeRect.y + allCircles[bestCircleIdx][1]);
		irisCenter.y = static_cast<int>(allCircles[bestCircleIdx][1]);
		radius = static_cast<int>(allCircles[bestCircleIdx][2]);
	}	// circles not empty

	// Fill the iris circle
	//CV_Assert(!irisMask.empty());	// check that we have created the iris mask
	//irisMask.create(hsvChannels[0].size());	// make sure the matrix is allocated
	//irisMask.setTo(0);	// clear the mask
	cv::Mat1b irisMask(eyeRect.size(), 0);
	cv::circle(irisMask, irisCenter, radius, cv::Scalar(255), -1);

	// Use the eye contour to clip the parts of the iris exceeding eye boundaries
	cv::Mat1b eyeMask(image.size(), 0);
	cv::fillConvexPoly(eyeMask, eyeContour, cv::Scalar(255));
	cv::bitwise_and(irisMask, eyeMask(eyeRect), irisMask);

	//// TEST!
	//cv::Mat imtest = hsvChannels[0].clone();
	//cv::circle(imtest, irisCenter, radius, cv::Scalar(0, 255, 0), 1);
	//cv::imshow("iris", imtest);
	//cv::waitKey();
	//cv::imshow("iris mask", irisMask);
	//cv::waitKey();

	// Create a new iris texture by mixing the original iris texture and the new color
	cv::Mat3b irisTexture(eyeRect.size(), cv::Vec3b(static_cast<uchar>(this->color[0]), static_cast<uchar>(this->color[1]), static_cast<uchar>(this->color[2])));
	cv::Mat3b imageROI = image(eyeRect);
	cv::seamlessClone(imageROI, irisTexture, irisMask, irisCenter, irisTexture, cv::MIXED_CLONE);

	//cv::imshow("test", this->iris);
	//cv::waitKey();


	// Prepare the input image ROI, the iris texture, and the iris mask for alpha matting

	cv::Mat3f irisMaskF, inputF, irisF;
	imageROI.convertTo(inputF, CV_32FC3, 1.0 / 255);
	irisTexture.convertTo(irisF, CV_32FC3, 1.0 / 255);

	double alpha = this->color[3] / 255;	// opacity 0..1
	cv::merge(std::vector<cv::Mat1b>{irisMask, irisMask, irisMask}, irisTexture);
	irisTexture.convertTo(irisMaskF, CV_32FC3, alpha * 1.0 / 255);

	// Blur the iris mask unless it's too small
	if (blur)
	{
		cv::erode(irisMaskF, irisMaskF, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
		//cv::blur(irisMaskF, irisMaskF, cv::Size(7, 7));
		cv::GaussianBlur(irisMaskF, irisMaskF, cv::Size(7, 7), 0, 0);
	}

	// Perform alpha matting

	cv::multiply(irisF, irisMaskF, irisF);
	//cv::imshow("iris", irisF);
	//cv::waitKey();

	irisMaskF.convertTo(irisMaskF, CV_32FC3, -1, +1);		// invert the mask
	cv::multiply(inputF, irisMaskF, inputF);

	inputF += irisF;
	//cv::imshow("colored", inputF);
	//cv::waitKey();

	inputF.convertTo(imageROI, CV_8UC3, 255.0);
}	// changeIrisColor_Overlaying
*/

//std::vector<cv::Vec3f> EyeColorFilter::detectIrisCircles(const std::vector<cv::Mat1b>& hsvChannels, const cv::Rect& eyeRect, 
//	int minRadius, int maxRadius) const
//{
//
//}	// detectIrisCircles

/*
void EyeColorFilter::createIrisMask(const std::vector<cv::Mat1b>& hsvChannels, const std::vector<cv::Point>& eyeContour,
	int minRadius, int maxRadius, cv::Point& irisCenter, cv::Mat1b& irisMask) const
{
	CV_Assert(!eyeContour.empty());
	CV_Assert(!hsvChannels.empty());
		
	// Use a slightly padded bounding rectange of the eye to restrict iris detection only in that region
	cv::Rect eyeRect = cv::boundingRect(eyeContour);
	eyeRect.x = std::max(0, eyeRect.x - maxRadius);
	eyeRect.y = std::max(0, eyeRect.y - maxRadius);
	eyeRect.width = std::min(hsvChannels[0].cols-eyeRect.x, eyeRect.width + 2*maxRadius);
	eyeRect.height = std::min(hsvChannels[0].rows-eyeRect.y, eyeRect.height + 2*maxRadius);
		
	std::vector<cv::Vec3f> allCircles;
	std::vector<cv::Vec3f> circles;
	for (const auto& channel : hsvChannels)
	{
		cv::Mat1b eyeGray = channel(eyeRect);

		//// TEST!
		//cv::imshow("eye gray", eyeGray);
		//cv::waitKey();

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
			circles.clear();
			// TODO: try HOUGH_GRADIENT_ALT (requires opencv 4.4)
			cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1, 1, threshold, 5, minRadius, maxRadius);
			//cv::HoughCircles(eyeGray, circles, cv::HOUGH_GRADIENT, 1.5, 1, threshold, 5, minRadius, maxRadius);
			
			// Filter the circles: the iris center must lie within the eye contour
			auto last = std::remove_if(std::execution::par_unseq, circles.begin(), circles.end(), 
						[&eyeContour, &eyeRect](const auto& circle)
						{
							return cv::pointPolygonTest(eyeContour, cv::Point2f(eyeRect.x + circle[0], eyeRect.y + circle[1]), false) < 0;
						});
			//allCircles.erase(last, allCircles.end());
			if (last != circles.begin())	// not empty vector
			{
				
				//// TEST!
				//cv::Mat imcircles;
				//cv::merge(std::vector<cv::Mat>{channel, channel, channel}, imcircles);
				////cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
				//for (const auto& circle : circles)
				//{
				//	cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
				//	//cv::circle(imcircles, cv::Point(circle[0], circle[1]), 1, cv::Scalar(255,0,0), -1);
				//}
				/////cv::imwrite("z:/circles.jpg", imcircles);
				//cv::imshow("circles " + std::to_string(t++), imcircles);
				//cv::waitKey();
				
				std::move(circles.begin(), last, std::back_inserter(allCircles));
				break;		
			}	// not empty 					

		}	// for threshold
		
	}	// for channels


	//// TEST!
	//cv::Mat imcircles;
	//cv::merge(std::vector<cv::Mat>{hsvChannels[0], hsvChannels[0], hsvChannels[0]}, imcircles);
	////cv::fillConvexPoly(imcircles, eyeContour, cv::Scalar(255, 255, 255));
	//for (const auto& circle : allCircles)
	//{
	//	cv::circle(imcircles, cv::Point(eyeRect.x+circle[0], eyeRect.y+circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);	// TEST!
	//	//cv::circle(imcircles, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(0, 255, 0), 1);
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

		auto bestCircleIdx = std::max_element(std::execution::par_unseq, ranks.begin(), ranks.end()) - ranks.begin();
		assert(bestCircleIdx >= 0);

		//for (std::size_t i = 0; i < allCircles.size(); ++i)	// TODO: try to use reduce here too
		//{
		//	//cv::Point2f center(allCircles[i][0], allCircles[i][1]);


		//	// The iris center must lie within the eye contour
		//	//assert(cv::pointPolygonTest(eyeContour, center, false) >= 0);

		//	ranks[i] = std::reduce(allCircles.begin(), allCircles.end(), 0.0, 
		//		[&center=cv::Point2f(allCircles[i][0], allCircles[i][1])](double vote, const auto& circle)
		//		{
		//			//double test = vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
		//			return vote + std::exp(-cv::norm(center - cv::Point2f(circle[0], circle[1])));
		//			//return test;
		//		});

		//	if (ranks[i] > ranks[bestCircleIdx])
		//		bestCircleIdx = i;
		//}	// for i

		// In case all the circles have very low ranks, it is better to go with our initial estimate 
		assert(ranks[bestCircleIdx] > 0);
		irisCenter.x = static_cast<int>(eyeRect.x + allCircles[bestCircleIdx][0]);
		irisCenter.y = static_cast<int>(eyeRect.y + allCircles[bestCircleIdx][1]);
		radius = static_cast<int>(allCircles[bestCircleIdx][2]);
	}	// circles not empty

	// Fill the iris circle
	//CV_Assert(!irisMask.empty());	// check that we have created the iris mask
	irisMask.create(hsvChannels[0].size());	// make sure the matrix is allocated
	irisMask.setTo(0);	// clear the mask
	cv::circle(irisMask, irisCenter, radius, cv::Scalar(255), -1);

	// Use the eye contour to clip the parts of the iris exceeding eye boundaries
	cv::Mat1b eyeMask(hsvChannels[0].size(), 0);
	cv::fillConvexPoly(eyeMask, eyeContour, cv::Scalar(255));
	cv::bitwise_and(irisMask, eyeMask, irisMask);

	//// TEST!
	//cv::Mat imtest = hsvChannels[0].clone();
	//cv::circle(imtest, irisCenter, radius, cv::Scalar(0, 255, 0), 1);
	//cv::imshow("iris", imtest);
	//cv::waitKey();
	//cv::imshow("iris mask", irisMask);
	//cv::waitKey();
}	// createIrisMask


void EyeColorFilter::changeIrisColor_Overlaying(cv::Mat3b& image, const cv::Mat1b& irisMask, const cv::Point& irisCenter, bool blur) const
{	
	cv::Mat3b iris(image.size(), cv::Vec3b(static_cast<uchar>(this->color[0]), static_cast<uchar>(this->color[1]), static_cast<uchar>(this->color[2])));
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

*/