#ifndef ABSTRACTLANDMARKDETECTOR_H
#define ABSTRACTLANDMARKDETECTOR_H

#include <opencv2/core.hpp>

//#include <vector>
#include <memory>



// An abstract parent class for landmark detectors

template <typename Landmarks>
class AbstractLandmarkDetector
{
public:
	virtual ~AbstractLandmarkDetector() = default;

	Landmarks detect(const cv::Mat& image) const { return detectLandmarks(image); }

	std::unique_ptr<AbstractLandmarkDetector<Landmarks>> clone() const { return createClone(); }

protected:
	AbstractLandmarkDetector() = default;

	AbstractLandmarkDetector(const AbstractLandmarkDetector&) = default;
	AbstractLandmarkDetector(AbstractLandmarkDetector&&) = default;

	AbstractLandmarkDetector& operator = (const AbstractLandmarkDetector&) = delete;
	AbstractLandmarkDetector& operator = (AbstractLandmarkDetector&&) = delete;

private:
	virtual std::unique_ptr<AbstractLandmarkDetector<Landmarks>> createClone() const = 0;
	virtual Landmarks detectLandmarks(const cv::Mat & image) const = 0;
};	// AbstractLandmarkDetector


#endif	// ABSTRACTLANDMARKDETECTOR_H