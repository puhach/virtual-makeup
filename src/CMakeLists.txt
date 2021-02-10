cmake_minimum_required(VERSION 3.0)

PROJECT(Virtual_Makeup)
SET(CMAKE_CXX_STANDARD 17)

######################## EDIT IF REQUIRED ####################
# ###Uncomment the line below and specify the path to OpenCV directory i.e. the path to the OpenCVConfig.cmake file. Check the examples given below.
#SET(OpenCV_DIR Enter-the-path-of-OpenCV-installation-on-your-system)
set(OpenCV_DIR /opt/opencv/4.5.1/installation/lib/cmake/opencv4)


################### OpenCV_DIR Examples  #####################

### MACOS : /usr/local/Cellar/opencv/3.3.1_1/share/OpenCV/

### UBUNTU : /usr/local/share/OpenCV/

### WINDOWS : C:\Users\yourname\Documents\opencv-3.3.1\build\install

##############################################################




if(MSVC)
	SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

find_package( OpenCV REQUIRED )


include(../dlib/dlib/cmake)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vimaku 
	main.cpp 
	abstractimagefilter.h 
	abstractimagefilter.cpp 
	abstractboxdetector.h 
	abstractlandmarkdetector.h 
	faciallandmarkdetector.h 
	faciallandmarkdetector.cpp
	faciallandmarkfilter.h
	faciallandmarkfilter.cpp
	eyecolorfilter.h
	eyecolorfilter.cpp
	lipstickcolorfilter.h
	lipstickcolorfilter.cpp
)

#[[
# GCC requires tbb to be linked in order to use execution policies
set(LINK_LIBS ${OpenCV_LIBS} dlib::dlib)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")    
    list(APPEND LINK_LIBS tbb)
endif()

#message(${LINK_LIBS})
target_link_libraries(vimaku ${LINK_LIBS})
]]
target_link_libraries(vimaku ${OpenCV_LIBS} dlib::dlib)
		
	
# Copy the shape predictor to the output directory
add_custom_command(TARGET vimaku
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     "${PROJECT_SOURCE_DIR}/../models/shape_predictor_68_face_landmarks.dat"
                     "$<TARGET_FILE_DIR:vimaku>")
	
# Copy test image files to the output directory
add_custom_command(TARGET vimaku
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   "${PROJECT_SOURCE_DIR}/../images"
                   "$<TARGET_FILE_DIR:vimaku>/images")

