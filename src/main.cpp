#include "faciallandmarkdetector.h"
#include "lipstickcolorfilter.h"
#include "eyecolorfilter.h"

#include <iostream>
#include <fstream>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

//using namespace std;
//using namespace cv;


// An auxiliary function for drawing facial landmarks
void drawLandmarks(cv::Mat& image, const std::vector<cv::Point>& landmarks)
{
	for (auto i = 0; i < landmarks.size(); ++i)
	{
		const auto& lm = landmarks.at(i);
		cv::Point center(lm.x, lm.y);
		cv::circle(image, center, 3, cv::Scalar(0, 255, 0), -1);
		cv::putText(image, std::to_string(i), center, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 0, 255), 1);
	}
}

// An auxiliary function for saving landmarks to a text file
void saveLandmarks(const std::string& fileName, const std::vector<cv::Point>& landmarks)
{
	std::ofstream ofs(fileName, std::ofstream::out);
	assert(ofs.good());

	ofs << landmarks.size() << std::endl;
	for (const auto lk : landmarks)
	{
		ofs << lk.x << " " << lk.y << std::endl;
	}

	assert(ofs.good());
}

// An auxiliary function for loading landmarks from a text file
std::vector<cv::Point> loadLandmarks(const std::string& fileName)
{
	std::ifstream ifs(fileName, std::ifstream::in);
	assert(ifs.good());

	size_t n;
	ifs >> n;

	std::vector<cv::Point> landmarks(n);
	for (size_t i = 0; i < n; ++i)
	{
		unsigned long x, y;
		ifs >> x >> y;

		//landmarks.emplace_back(x, y);
		landmarks[i].x = x;
		landmarks[i].y = y;
	}

	return landmarks;
}


// Converts hexadecimal RGB/RGBA strings like "FF12BB" and "FF12BB33" to OpenCV Scalar. The first 6 digits define the values 
// of R, G, B components. The last pair of digits, interpreted as a hexadecimal number, specifies the alpha channel of the color, 
// where 00 represents a fully transparent color and FF represents a fully opaque color. The optional swapRB parameter determines 
// whether the red and blue components must be swapped.
cv::Scalar strToColor(std::string_view hexRGBA, bool swapRB = true)
{
	if (hexRGBA.size() != 6 && hexRGBA.size() != 8)
		throw std::invalid_argument(std::string("The specified color string has invalid length: ").append(hexRGBA));

	std::string s(8, 'F');
	std::transform(hexRGBA.begin(), hexRGBA.end(), s.begin(), [](const char& c)
	{
		return std::toupper(c);		
	});

	cv::Scalar colorRGBA;
	for (unsigned short i = 0; i < s.size(); ++i)
	{
		char c = s[i];
		if (c >= '0' && c <= '9')
			c -= '0';
		else if (c >= 'A' && c <= 'F')
			c = 10 + c - 'A';
		else 
			throw std::invalid_argument("Invalid character. Only hex characters (0..9 and A..F) are allowed.");

		colorRGBA[i / 2] = colorRGBA[i / 2] * 16 + c;
	}

	if (swapRB)
		std::swap(colorRGBA[0], colorRGBA[2]);	// swap red and blue components

	return colorRGBA;
}

// Prints the help message describing how to use the program
void printUsage()
{
	std::cout << "Usage: vimaku [-h]"
				 " --input=<input image file>" 				      
				 " [--lipstick_color=<the new lipstick color in the RRGGBBAA notation>]"
				 " [--eye_color=<the new color of the eyes in the RRGGBBAA notation>]"
                 " [--output=<output image file>]" << std::endl;
}


int main(int argc, char* argv[])
{
    try
    {
        static const cv::String keys =
			"{help h usage ?   |         | Print the help message  }"			
			"{input            |<none>   | The input image file }"
			"{lipstick_color   |         | The new lipstick color in RRGGBBAA notation }"
			"{eye_color        |         | The new iris color in RRGGBBAA notation  }"
			"{output           |         | If not empty, specifies the file where the output image will be saved to }";
		
		cv::CommandLineParser parser(argc, argv, keys);
		parser.about("Glassify\n(c) Yaroslav Pugach");

		if (parser.has("help"))
		{
			printUsage();
			return 0;
		}

		std::string inputFile = parser.get<std::string>("input");
		std::string lipstickColor = parser.get<std::string>("lipstick_color");
		std::string eyeColor = parser.get<std::string>("eye_color");
		std::string outputFile = parser.get<std::string>("output");
		
		if (!parser.check())
		{
			parser.printErrors();
			printUsage();
			return -1;
		}
		
		// Load the input image
		cv::Mat imSrc = cv::imread(inputFile, cv::IMREAD_COLOR);
        CV_Assert(!imSrc.empty());
        
        // Upscale the input image if it is too small        
        cv::Mat imSrcResized;        
        if (double imScale = std::max(imSrc.rows, imSrc.cols) / 600.0; imScale < 1.0)
            cv::resize(imSrc, imSrcResized, cv::Size(), 1 / imScale, 1 / imScale, cv::INTER_CUBIC);
        else
            imSrc.copyTo(imSrcResized);
        
        
        // Detect the landmarks
        auto facialLandmarkDetector = std::make_shared<FacialLandmarkDetector>(0.5);    
        auto landmarks = facialLandmarkDetector->detect(imSrcResized);
                
        // Build the filter pipeline
        
        std::vector<std::unique_ptr<FacialLandmarkFilter>> filters;
        
        if (!lipstickColor.empty())
            filters.push_back(std::make_unique<LipstickColorFilter>(facialLandmarkDetector, strToColor(lipstickColor)));
        
        if (!eyeColor.empty())
            filters.push_back(std::make_unique<EyeColorFilter>(facialLandmarkDetector, strToColor(eyeColor)));
        
        // More filters can be added here
        
        
        // Apply filters to the input image
        cv::Mat imOut = imSrcResized;
        for (const auto &filter : filters)
        {
            //filter->apply(imOut, landmarks, imOut);
            filter->applyInPlace(imOut, landmarks);
        }

        // Restore the original image size        
        if (imOut.size() != imSrc.size())
            cv::resize(imOut, imOut, imSrc.size(), 0, 0, cv::INTER_AREA);
        
        // Place the original and the output images next to each other
        cv::Mat imCombined;
        cv::hconcat(imSrc, imOut, imCombined);
        
        // Downscale the concatenated image if it is too large
        double scale = std::max(imCombined.rows, imCombined.cols) / 1000.0;
        if (scale > 1.0)
            cv::resize(imCombined, imCombined, cv::Size(0, 0), 1 / scale, 1 / scale, cv::INTER_AREA);

        // Show the combined image to the user 
        cv::imshow(inputFile, imCombined);
        cv::waitKey();

        // Save the output if needed
        if (!outputFile.empty() && !cv::imwrite(outputFile, imOut))
            throw std::runtime_error("Failed to save the output image to " + outputFile);
    }   // try
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }   // catch

	return 0;
}
