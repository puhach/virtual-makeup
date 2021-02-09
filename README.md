# Virtual Makeup

![Virtual Makeup](./assets/cover.jpg)

With a virtual makeup try-on tool implemented in this project you can:

* Apply lipstick

* Change eye color

More features will be implemented later.

The usage details are described below.

## Set Up

It is assumed that OpenCV 4.x, C++17 compiler, and cmake 3.0 or newer are installed on the system.

### Project structure

The project has the following directory structure:
```
│   .gitignore
│   README.md
│   
├───assets
│       
├───dlib
│               
├───images
│       
├───models
│       shape_predictor_68_face_landmarks.dat
│       
├───src
│   │   abstractboxdetector.h
│   │   abstractimagefilter.cpp
│   │   abstractimagefilter.h
│   │   abstractlandmarkdetector.h
│   │   CMakeLists.txt
│   │   eyecolorfilter.cpp
│   │   eyecolorfilter.h
│   │   faciallandmarkdetector.cpp
│   │   faciallandmarkdetector.h
│   │   faciallandmarkfilter.cpp
│   │   faciallandmarkfilter.h
│   │   lipstickcolorfilter.cpp
│   │   lipstickcolorfilter.h
│   │   main.cpp
│   │   
│   └───build
│                           
└───writeup
        eye_color.docx
        lipstick.docx
        
```

### Download Dlib

Dlib C++ library can be downloaded from dlib.net. The project was tested with Dlib 19.21.

After downloading extract the archive to the `dlib` folder in the project root (see the directory structure above). 

### Download the Facial Landmark Detection Model

Download the 68 facial landmark predictor from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2). Just in case the URL doesn't work, consult [this](https://github.com/davisking/dlib-models#shape_predictor_68_face_landmarksdatbz2) page.

Extract the predictor to the `models` folder in the project root (see the directory structure above).

### Specify OpenCV_DIR in CMakeLists

Open CMakeLists.txt and set the correct OpenCV directory in the following line:

```
set(OpenCV_DIR /opt/opencv/4.5.1/installation/lib/cmake/opencv4)
```

Depending on the platform and the way OpenCV was installed, it may be needed to provide the path to cmake files explicitly. On my KUbuntu 20.04 after building OpenCV 4.5.1 from sources the working `OpenCV_DIR` looks like <OpenCV installation path>/lib/cmake/opencv4. On Windows 8.1 after installing a binary distribution of OpenCV 4.2.0 it is C:\OpenCV\build.


### Build the Project

In the `src` folder create the `build` directory unless it already exists. Then from the terminal run the following:

```
cd build
cmake ..
```

This should generate the build files. When it's done, compile the code:

```
cmake --build . --config Release
```


## Usage

The program has to be run from the command line. It takes in the path to the image containing the warped document and several optional parameters: 

```
doscan  --input=<input image file>
		[--output=<output file>]
		[--view_invariant=<true or false>]
		[--width=<a positive integer or zero>]
		[--height=<a positive integer or zero>]
		[--aspect_ratio=<a positive float>]
		[--paper_detector=<1 - Ithresh, 2 - Savaldo>]
		[--threshold_levels=<integer (1..255)>]
		[--min_area_pct=<float (0..max_area_pct)>]
		[--max_area_pct=<float (min_area_pct..1)>]
		[--approx_accuracy_pct=<float (0..1)>]"
	 	[--help]
```

Parameter    | Meaning 
------------ | --------------------------------------
help, ? | Prints the help message.
input | The file path of the image to be rectified.
output | If not empty, specifies the output file path where the rectified image will be saved to.
view_invariant | Determines whether the document's aspect ratio should be treated as view-invariant (true by default). 
width | The rectified document's width (if zero, it is deduced from the height and the aspect ratio). Defaults to 500. 
height | The rectified document's height (if zero, it is deduced from the width and the aspect ratio). Defaults to 0. 
aspect_ratio | The rectified document's aspect ratio (unused if both width and height are specified). Defaults to 0.7071.
paper_detector | The algorithm to be used for paper sheet detection (1 - Ithresh, 2 - Savaldo). Defaults to 1.
threshold_levels | The number of threshold levels for the Ithresh paper sheet detector. Default value is 15.
min_area_pct | The minimal fraction of the original image that the paper sheet must occupy to be considered for detection (0..max_area_pct). Default value is 0.5.
max_area_pct | The maximal fraction of the original image that the paper sheet can occupy to be considered for detection (min_area_pct..1). Default value is 0.99.
approx_accuracy_pct | The accuracy of contour approximation with respect to the contour length (0..1). Default value is 0.02.


Sample usage (linux):
```
./doscan --input=../images/dragon-medium.jpg 
```

The application will detect the document using default parameters. The user may adjust the document boundaries by dragging the vertices. 

![document detection](./assets/detection.jpg)

To rectify the document press any key. 

![rectified document](./assets/rectified.jpg)

Pressing *Escape* will quit the application.
