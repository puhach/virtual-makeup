#ifndef ABSTRACTIMAGEFILTER_H
#define ABSTRACTIMAGEFILTER_H


#include <opencv2/core.hpp>

#include <memory>


// An abstract parent class for image filters

class AbstractImageFilter
{
public:
	virtual ~AbstractImageFilter() = default;

	cv::Mat apply(const cv::Mat& image) const;	// always allocates a new matrix to store the output
	void apply(const cv::Mat& image, cv::Mat& out) const;	// may be useful if the output matrix of the matching type has already been allocated
		

	// C.67: A base class should suppress copying, and provide a virtual clone instead if "copying" is desired
	virtual std::unique_ptr<AbstractImageFilter> clone() const = 0;

protected:
	AbstractImageFilter() = default;
	AbstractImageFilter(const AbstractImageFilter&) = default;
	AbstractImageFilter(AbstractImageFilter&&) = default;

	// Possible solutions to assignment of polymorphic classes are proposed here:
	// https://www.fluentcpp.com/2020/05/22/how-to-assign-derived-classes-in-cpp/
	AbstractImageFilter& operator = (const AbstractImageFilter&) = delete;
	AbstractImageFilter& operator = (AbstractImageFilter&&) = delete;

	virtual void applyInPlace(cv::Mat& image) const = 0;	// stores the result into the same matrix as the input
};	// AbstractImageFilter



#endif	// ABSTRACTIMAGEFILTER_H