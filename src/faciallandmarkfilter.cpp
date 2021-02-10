#include "faciallandmarkfilter.h"

cv::Mat FacialLandmarkFilter::apply(const cv::Mat& image, const std::vector<cv::Point>& landmarks) const
{
	cv::Mat out = image.clone();
	applyInPlace(out, landmarks);
	return out;
}


cv::Mat FacialLandmarkFilter::apply(const cv::Mat& image, std::vector<cv::Point>&& landmarks) const
{
	cv::Mat out = image.clone();
	applyInPlace(out, std::move(landmarks));
	return out;
}

void FacialLandmarkFilter::apply(const cv::Mat& image, const std::vector<cv::Point>& landmarks, cv::Mat& out) const
{
	image.copyTo(out);
	return applyInPlace(out, landmarks);
}

void FacialLandmarkFilter::apply(const cv::Mat& image, std::vector<cv::Point>&& landmarks, cv::Mat& out) const
{
	image.copyTo(out);
	return applyInPlace(out, std::move(landmarks));
}

void FacialLandmarkFilter::applyInPlace(cv::Mat& image, const std::vector<cv::Point>& landmarks) const
{
	FacialLandmarkFilter::ExistingLandmarksRAII raii(this->landmarks);
	this->landmarks = landmarks;
	modify(image);
}

void FacialLandmarkFilter::applyInPlace(cv::Mat& image, std::vector<cv::Point>&& landmarks) const
{
	FacialLandmarkFilter::ExistingLandmarksRAII raii(this->landmarks);
	this->landmarks = std::move(landmarks);
	modify(image);
}


std::vector<cv::Point> FacialLandmarkFilter::grabLandmarks(const cv::Mat& image) const
{
	if (this->landmarks.empty())
	{
		return this->landmarkDetector->detect(image);
	}
	else
	{
		std::vector<cv::Point> buf;		// noexcept
		this->landmarks.swap(buf);	// make landmarks empty (noexcept)
		return buf;
	}
}	// grabLandmarks