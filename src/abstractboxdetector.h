#ifndef ABSTRACTBOXDETECTOR_H
#define ABSTRACTBOXDETECTOR_H


#include <opencv2/core.hpp>

#include <memory>



// An abstract parent class for single-class bounding box detectors

class AbstractBoxDetector
{
public:

	virtual ~AbstractBoxDetector() = default;
	
	cv::Rect detect(const cv::Mat& image) const { return detectObject(image); }

	std::unique_ptr<AbstractBoxDetector> clone() const { return createClone(); }	

protected:

	AbstractBoxDetector() = default;
	
	// Restrict copy/move operations since this is a polymorphic type

	AbstractBoxDetector(const AbstractBoxDetector&) = default;
	AbstractBoxDetector(AbstractBoxDetector&&) = default;

	AbstractBoxDetector& operator = (const AbstractBoxDetector&) = delete;
	AbstractBoxDetector& operator = (AbstractBoxDetector&&) = delete;

private:	

	virtual std::unique_ptr<AbstractBoxDetector> createClone() const = 0;
	virtual cv::Rect detectObject(const cv::Mat &image) const = 0;
};	// AbstractBoxDetector




#endif	// ABSTRACTBOXDETECTOR_H