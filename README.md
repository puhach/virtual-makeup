# Virtual Makeup

![Virtual Makeup](./assets/green_eyes_red_lipstick.jpg)

With a virtual makeup try-on tool implemented in this project you can:

* Apply lipstick

* Change eye color

More features will be added in time.

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

The program has to be run from the command line. It takes in the path to the image file and several optional parameters: 

```
vimaku  --input=<input image file>
	[--output=<output image file>]
	[--lipstick_color=<The new lipstick color>]
	[--eye_color=<The new eye color>]		
	[--help]
```

Parameter    | Meaning 
------------ | --------------------------------------
help, ? | Prints the help message.
input | The input image file.
output | If not empty, specifies the output file where the output image will be saved to.
lipstick_color | The new lipstick color in RRGGBBAA notation. If empty, the lipstick filter will not be applied.
eye_color | The new iris color in RRGGBBAA notation. If empty, the eye color filter will not be applied.

The RRGGBBAA color notation is similar to the one used in CSS, but you don't need to prepend it with the hash sign. The first 6 digits define the values 
of R, G, B components. The last pair of digits, interpreted as a hexadecimal number, specifies the alpha channel of the color, where 00 represents a fully transparent color and FF represents a fully opaque color. In case the alpha component is omitted, it is assumed to be FF. For example, FF000070 specifies a pure semi-transparent red color. FF0000 and ff0000ff are identical and specify a fully opaque red color.

Applying lipstick example:
```
./vimaku --input=./images/girl-no-makeup.jpg --lipstick_color=FF000050 
```

This will add a mild red lipstick effect:

![Applying lipstick](./assets/lipstick.jpg)

For realistic look I recommend alpha values from 30 to 70.


Changing eye color example:
```
./vimaku --input=./images/girl7.png --eye_color=2E1902CC  
```

This will change the iris color to brown:
![Brown eyes](./assets/eye_color_brown.jpg)

Recommended values of alpha for the eye color filter depend on the original iris color and the intensity of the new color. When eyes are originally light, the alpha values should normally be less than 70.  

Filters can be applied together:
```
./vimaku --input=./images/girl5_small.jpg --eye_color=4b724882 --lipstick_color=ff7f5050 --ouput=out.jpg  
```

This will change the iris color to blue and the lipstick color to orange. The output image will be saved to out.jpg. 

![Blue eyes and orange lipstick](./assets/blue_eyes_and_orange_lipstick.jpg)


## Credits

Images have been downloaded from pinterest.com.
