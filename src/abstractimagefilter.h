#ifndef ABSTRACTIMAGEFILTER_H
#define ABSTRACTIMAGEFILTER_H


#include <opencv2/core.hpp>

#include <memory>


// An abstract parent class for image filters

class AbstractImageFilter
{
public:

	virtual ~AbstractImageFilter() = default;

	// The following methods apply this filter to the input image. Several overloads are provided for convenience.

	cv::Mat apply(const cv::Mat& image) const;	// always allocates a new matrix to store the output
	void apply(const cv::Mat& image, cv::Mat& out) const;	// may be useful if the output matrix of the matching type has already been allocated
	
	void applyInPlace(cv::Mat& image) const;	// stores the result into the same matrix as the input

	// C.67: A base class should suppress copying, and provide a virtual clone instead if "copying" is desired
	std::unique_ptr<AbstractImageFilter> clone() const { return createClone(); }

protected:

	AbstractImageFilter() = default;
	AbstractImageFilter(const AbstractImageFilter&) = default;
	AbstractImageFilter(AbstractImageFilter&&) = default;

	// Possible solutions to assignment of polymorphic classes are proposed here:
	// https://www.fluentcpp.com/2020/05/22/how-to-assign-derived-classes-in-cpp/
	AbstractImageFilter& operator = (const AbstractImageFilter&) = delete;
	AbstractImageFilter& operator = (AbstractImageFilter&&) = delete;

private:

	// This private function is called by multiple public overloads to apply the filter in-place.
	// It is virtual and can be overridden in descendants to alter filter behavior.
	virtual void modify(cv::Mat& image) const = 0;	

	// Returns a smart pointer which owns and manages a copy of this instance
	virtual std::unique_ptr<AbstractImageFilter> createClone() const = 0;
};	// AbstractImageFilter



#endif	// ABSTRACTIMAGEFILTER_H