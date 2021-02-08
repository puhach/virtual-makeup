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

using namespace std;
using namespace cv;


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
vector<Point> loadLandmarks(const std::string& fileName)
{
	std::ifstream ifs(fileName, std::ifstream::in);
	assert(ifs.good());

	size_t n;
	ifs >> n;

	vector<Point> landmarks(n);
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
// where 00 represents a fully transparent color and FF represent a fully opaque color. The optional swapRB parameter determines 
// whether the red and blue components must be swapped.
cv::Scalar strToColor(std::string_view hexRGBA, bool swapRB = true)
{
	if (hexRGBA.size() != 6 && hexRGBA.size() != 8)
		throw invalid_argument(string("The specified color string has invalid length: ").append(hexRGBA));

	string s(8, 'F');
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
			throw invalid_argument("Invalid character. Only hex characters (0..9 and A..F) are allowed.");

		colorRGBA[i / 2] = colorRGBA[i / 2] * 16 + c;
	}

	if (swapRB)
		std::swap(colorRGBA[0], colorRGBA[2]);	// swap red and blue components

	return colorRGBA;
}


int main(int argc, char* argv[])
{
	
	try
	{
		//vector<string> inputFiles{ "./girl-no-makeup.jpg", "./girl2.png", "./girl3.jpg", "./girl4.jpg", "./girl5_small.jpg", "./girl6.jpg", "./girl7.png", "z:/boy1.jpg" };
		vector<string> inputFiles{ "images/girl-no-makeup.jpg", "images/girl2.png", "images/girl3.jpg", "images/girl4.jpg", "images/girl5_small.jpg", "images/girl6.jpg", "images/girl7.png", "images/boy1.jpg" };
		vector<string> landmarkFiles{ "z:/girl1.txt", "z:/girl2.txt", "z:/girl3.txt", "z:/girl4.txt", "z:/girl5.txt", "z:/girl6.txt", "z:/girl7.txt", "z:/boy1.txt" };

		auto facialLandmarkDetector = make_shared<FacialLandmarkDetector>(0.5);		// TODO: replace the scaling factor with resizeHeight
		unique_ptr<EyeColorFilter> eyeColorFilter = make_unique<EyeColorFilter>(facialLandmarkDetector);
		unique_ptr<LipstickColorFilter> lipstickColorFilter = make_unique<LipstickColorFilter>(facialLandmarkDetector);

		//namedWindow("test", cv::WINDOW_AUTOSIZE);

		for (int i = 0; i < inputFiles.size(); ++i)
		{
			Mat im = imread(inputFiles[i], IMREAD_COLOR);
			CV_Assert(!im.empty());

			cv::Size imSizeOriginal = im.size();
			double imScale = max(im.rows, im.cols) / 600.0;
			if (imScale < 1.0)
				cv::resize(im, im, cv::Size(), 1 / imScale, 1 / imScale, cv::INTER_CUBIC);

			//auto landmarks = facialLandmarkDetector->detect(im);
			//saveLandmarks(landmarkFiles[i], landmarks);
			//cv::Mat out = eyeColorFilter->apply(im, landmarks);
			auto landmarks = loadLandmarks(landmarkFiles[i]);
			cv::Mat out = eyeColorFilter->apply(im, landmarks);
			lipstickColorFilter->applyInPlace(out, landmarks);

			cv::Mat imCombined;
			cv::hconcat(im, out, imCombined);
			cv::imwrite("z:/result" + std::to_string(i + 1) + ".jpg", imCombined);

			double scale = max(imCombined.rows, imCombined.cols) / 1000.0;
			if (scale > 1.0)
				cv::resize(imCombined, imCombined, cv::Size(0, 0), 1 / scale, 1 / scale, cv::INTER_LINEAR);

			cv::imshow("test", imCombined);
			cv::waitKey();

			if (out.size() != imSizeOriginal)
				cv::resize(out, out, imSizeOriginal, 0, 0, cv::INTER_AREA);

		}	// for i
	} // catch
	catch (const std::exception& e)
	{
		cerr << e.what() << endl;
		return -1;
	}
	

	/*
	vector<string> inputFiles{"./girl-no-makeup.jpg", "./girl2.png", "./girl3.jpg", "./girl4.jpg", "./girl5.png", "./girl6.jpg", "./girl7.png", "z:/boy1.jpg"};
	vector<string> landmarkFiles{"z:/landmarks.txt", "z:/landmarks2.txt", "z:/landmarks3.txt", "z:/landmarks4.txt", "z:/landmarks5.txt", "z:/landmarks6.txt", "z:/landmarks7.txt", "z:/boy1.txt"};

	//auto facialLandmarkDetector = make_shared<FacialLandmarkDetector>(0.5);		// TODO: replace the scaling factor with resizeHeight
	unique_ptr<EyeColorFilter> eyeColorFilter = make_unique<EyeColorFilter>(nullptr);

	//namedWindow("test", cv::WINDOW_AUTOSIZE);

	for (int i = 0; i < inputFiles.size(); ++i)
	{
		Mat im = imread(inputFiles[i], IMREAD_COLOR);
		CV_Assert(!im.empty());

		auto landmarks = loadLandmarks(landmarkFiles[i]);
		cv::Mat out = eyeColorFilter->apply(im, landmarks);

		cv::hconcat(im, out, out);
		cv::imwrite("z:/result" + std::to_string(i+1) + ".jpg", out);

		double scale = max(out.rows, out.cols) / 1000.0;
		if (scale > 1.0)
			cv::resize(out, out, cv::Size(0,0), 1/scale, 1/scale, cv::INTER_LINEAR);

		cv::imshow("test", out);
		cv::waitKey();
	}
	*/

	/*
	//Mat im = imread("./girl-no-makeup.jpg", IMREAD_COLOR);
	Mat im = imread("./girl2.png", IMREAD_COLOR);
	//Mat im = imread("./girl5_small.jpg", IMREAD_COLOR);
	//Mat im = imread("./girl6_large.jpg", IMREAD_COLOR);
	//Mat im = imread("./girl5.png", IMREAD_COLOR);
	//Mat im = imread("./girl7.png", IMREAD_COLOR);
	//Mat im = imread("./boy1.jpg", IMREAD_COLOR);
	CV_Assert(!im.empty());

	//cv::resize(im, im, Size(), 2, 2);		// TEST!
	//cv::imwrite("z:/girl5_2x.jpg", im);

	//unique_ptr<FacialLandmarkDetector<std::vector<cv::Point>>> facialLandmarkDetector = make_unique<FacialLandmarkDetector<std::vector<cv::Point>>>();
	//FacialLandmarkDetector<std::vector<cv::Point>> facialLandmarkDetector(0.5);
	//FacialLandmarkDetector facialLandmarkDetector(0.5);
	//auto landmarks1 = facialLandmarkDetector.detect(im);
	auto facialLandmarkDetector = make_shared<FacialLandmarkDetector>(0.5);		// TODO: replace the scaling factor with resizeHeight
	//auto landmarks = facialLandmarkDetector->detect(im);

	//saveLandmarks("z:/boy1.txt", landmarks);
	//saveLandmarks("z:/girl5_small.txt", landmarks);
	//auto landmarks = loadLandmarks("z:/girl5_small.txt");
	//auto landmarks = loadLandmarks("z:/girl6_large.txt");
	auto landmarks = loadLandmarks("z:/landmarks2.txt");
	//auto landmarks = loadLandmarks("z:/boy1.txt");
	//drawLandmarks(im, landmarks);
	//cv::imshow("test", im);
	//cv::waitKey();
	//imwrite("z:/landmarks.jpg", im);
	
	unique_ptr<LipstickColorFilter> lipstickColorFilter = make_unique<LipstickColorFilter>(facialLandmarkDetector);
	//lipstickColorFilter->setLandmarks(landmarks);	// optional
	//auto out = lipstickColorFilter->apply(im, std::move(landmarks));
	lipstickColorFilter->setColor(strToColor("Ff023050"));
	
	unique_ptr<EyeColorFilter> eyeColorFilter = make_unique<EyeColorFilter>(facialLandmarkDetector);
	//eyeColorFilter->setColor();
	
	//cv::Mat result = lipstickColorFilter->apply(im, landmarks);
	//lipstickColorFilter->apply()

	//cv::Mat result = eyeColorFilter->apply(im);
	cv::Mat result = eyeColorFilter->apply(im, landmarks);
	//eyeColorFilter->applyInPlace(im, landmarks);
	//eyeColorFilter->applyInPlace(result, landmarks);

	imshow("test", result);
	waitKey();
	cv::hconcat(im, result, result);
	imwrite("z:/result.jpg", result);
	*/

	return 0;
}
