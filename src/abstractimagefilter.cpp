#include "abstractimagefilter.h"




cv::Mat AbstractImageFilter::apply(const cv::Mat& image) const
{
	cv::Mat imageCopy = image.clone();
	modify(imageCopy);	// virtual call
	return imageCopy;
}

void AbstractImageFilter::apply(const cv::Mat& image, cv::Mat& out) const
{
	image.copyTo(out);
	modify(out);
}

void AbstractImageFilter::applyInPlace(cv::Mat& image) const
{
	modify(image);
}