cmake_minimum_required(VERSION 3.0)

project(Look3D)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/External/Libs/opencv)
#
# Find includes in corresponding build directories
#
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_INSTALL_PREFIX C:/Program\ Files/360\ Vision CACHE FILEPATH "" FORCE)

#
# Qt
# 
find_package(Qt5Widgets REQUIRED)
#find_package(Qt5Network REQUIRED)
find_package(Qt5Qml REQUIRED)
find_package(Qt5Quick REQUIRED)
find_package(Qt5Xml REQUIRED)
#
# QtOpenGL
#
find_package(Qt5OpenGL REQUIRED)

#set(QT5P7_API_DIR ${CMAKE_SOURCE_DIR}/External/Libs/qt5.7 CACHE PATH "")
#set(QT5P7_INC_DIR ${QT5P7_API_DIR}/include  CACHE PATH "")
#set(QT5P7_INC_GUI_DIR ${QT5P7_API_DIR}/include/qtgui  CACHE PATH "")
#set(QT5P7_INC_CORE_DIR ${QT5P7_API_DIR}/include/qtcore  CACHE PATH "")
#set(QT5P7_INC_QML_DIR ${QT5P7_API_DIR}/include/qtqml  CACHE PATH "")
#set(QT5P7_INC_WIDGETS_DIR ${QT5P7_API_DIR}/include/qtwidgets  CACHE PATH "")
#set(QT5P7_INC_QUICK_DIR ${QT5P7_API_DIR}/include/qtquick  CACHE PATH "")
#set(QT5P7_LIB_DIR ${QT5P7_API_DIR}/lib CACHE PATH "")
#set(QT5P7_LIB_OpenGL_LIBS optimized ${QT5P7_LIB_DIR}/Qt5OpenGL.lib debug ${QT5P7_LIB_DIR}/Qt5OpenGLd.lib)
#set(QT5P7_LIB_Core_LIBS optimized ${QT5P7_LIB_DIR}/Qt5Core.lib debug ${QT5P7_LIB_DIR}/Qt5Cored.lib)
#set(QT5P7_LIB_Quick_LIBS optimized ${QT5P7_LIB_DIR}/Qt5Quick.lib debug ${QT5P7_LIB_DIR}/Qt5Quickd.lib)
#set(QT5P7_LIB_Xml_LIBS optimized ${QT5P7_LIB_DIR}/Qt5Xml.lib debug ${QT5P7_LIB_DIR}/Qt5Xmld.lib)
#set(QT5P7_LIB_Qml_LIBS optimized ${QT5P7_LIB_DIR}/Qt5Qml.lib debug ${QT5P7_LIB_DIR}/Qt5Qmld.lib)
#set(QT5P7_LIB_Widgets_LIBS optimized ${QT5P7_LIB_DIR}/Qt5Widgets.lib debug ${QT5P7_LIB_DIR}/Qt5Widgetsd.lib)
#set(QT5P7_LIB_gui_LIBS optimized ${QT5P7_LIB_DIR}/Qt5Gui.lib debug ${QT5P7_LIB_DIR}/Qt5Guid.lib)
#set(QT5P7_LIB_Network_LIBS optimized ${QT5P7_LIB_DIR}/Qt5Network.lib debug ${QT5P7_LIB_DIR}/Qt5Networkd.lib)

#
# default OpenCV
#
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/External/Libs/opencv CACHE PATH "")
find_package(OpenCV REQUIRED)
set(OpenCV_INC_DIR ${OpenCV_DIR}/include CACHE PATH "")

#
# OpenGL
#
find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message(ERROR "OPENGL not found!")
endif(NOT OPENGL_FOUND)


#
#CUDA
#
find_package(CUDA)

if(CUDA_FOUND)
	add_definitions(/DUSE_CUDA)
else()
	message("CUDA not found. Project will be configured and generated only by OpenGL")
endif()
	

set(CMAKE_CONFIGURATION_TYPES 
	Debug 
	Release CACHE STRING "" FORCE)

#
# ffmpeg
#
set(FFMPEG_API_DIR ${CMAKE_SOURCE_DIR}/External/Libs/ffmpeg CACHE PATH "")
set(FFMPEG_INC_DIR ${FFMPEG_API_DIR}/include  CACHE PATH "")
set(FFMPEG_LIB_DIR ${FFMPEG_API_DIR}/windows/lib/x64 CACHE PATH "")
set(FFMPEG_BIN_DIR ${FFMPEG_API_DIR}/windows/bin/x64 CACHE PATH "")
find_library(FFMPEG_AVCODEC_LIBS avcodec ${FFMPEG_LIB_DIR})
find_library(FFMPEG_AVFORMAT_LIBS avformat ${FFMPEG_LIB_DIR})
find_library(FFMPEG_AVUTIL_LIBS avutil ${FFMPEG_LIB_DIR})
find_library(FFMPEG_AVDEVICE_LIBS avdevice ${FFMPEG_LIB_DIR})
find_library(FFMPEG_SWSCALE_LIBS swscale ${FFMPEG_LIB_DIR})
find_library(FFMPEG_SWRESAMPLE_LIBS swresample ${FFMPEG_LIB_DIR})

#
#libzip
#
set(LIBZIP_API_DIR ${CMAKE_SOURCE_DIR}/External/Libs/libzip CACHE PATH "")
set(LIBZIP_INC_DIR ${LIBZIP_API_DIR}/include CACHE PATH "")
set(LIBZIP_LIB_DIR ${LIBZIP_API_DIR}/x64/VS2015 CACHE PATH "")
#find_library(LIBZIP_ZIP_LIBS zip ${LIBZIP_LIB_DIR})
#find_library(LIBZIP_ZIPD_LIBS zipD ${LIBZIP_LIB_DIR})
set(LIBZIP_ZIP_LIBS optimized ${LIBZIP_LIB_DIR}/zip.lib debug ${LIBZIP_LIB_DIR}/zipD.lib)

# Oculus VR SDK dependency 
#
set(OCULUS_1_8_0 ON)
if(OCULUS_1_8_0)
message("Oculus SDK version 1.8.0")
set(OVR_API_DIR ${CMAKE_SOURCE_DIR}/External/Libs/OculusSDK/1.8.0/LibOVR CACHE PATH "")
set(OVR_INC_DIR ${OVR_API_DIR}/Include CACHE PATH "")
set(OVR_LIB_DIR ${OVR_API_DIR}/Lib/Windows/x64/Release/VS2015 CACHE PATH "")

set(OVRKERNEL_API_DIR ${CMAKE_SOURCE_DIR}/External/Libs/OculusSDK/1.8.0/LibOVRKernel CACHE PATH "")
set(OVRKERNEL_INC_DIR ${OVRKERNEL_API_DIR}/Src CACHE PATH "")
else()
message("Oculus SDK version 0.8.0")
set(OVR_API_DIR ${CMAKE_SOURCE_DIR}/External/Libs/OculusSDK/LibOVR CACHE PATH "")
set(OVR_INC_DIR ${OVR_API_DIR}/Include CACHE PATH "")
set(OVR_LIB_DIR ${OVR_API_DIR}/Lib/Windows/x64/VS2015 CACHE PATH "")
endif()
find_library(OVR_LIBS LibOVR ${OVR_LIB_DIR})

#
# Eigen Library dependency
#
set(EIGEN_INC_DIR ${CMAKE_SOURCE_DIR}/External/Libs/Eigen CACHE PATH "")

#
# Libsourcey Library dependency
#
add_definitions(/DNOMINMAX)
add_definitions(/DWEBRTC_WIN)

set(LIBSOURCEY_API_DIR ${CMAKE_SOURCE_DIR}/External/Libs/libsourcey CACHE PATH "")
set(LIBSOURCEY_INC_DIR ${LIBSOURCEY_API_DIR}/include  CACHE PATH "")
set(LIBSOURCEY_LIB_DIR ${LIBSOURCEY_API_DIR}/lib CACHE PATH "")
#find_library(LIBSOURCEY_SCY_ARCHO_LIBS scy_archo ${LIBSOURCEY_LIB_DIR} )
#find_library(LIBSOURCEY_SCY_ARCHOD_LIBS scy_archod ${LIBSOURCEY_LIB_DIR})

set(LIBSOURCEY_SCY_ARCHO_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_archo.lib debug ${LIBSOURCEY_LIB_DIR}/scy_archod.lib)
set(LIBSOURCEY_SCY_AV_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_av.lib debug ${LIBSOURCEY_LIB_DIR}/scy_avd.lib)
set(LIBSOURCEY_SCY_BASE_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_base.lib debug ${LIBSOURCEY_LIB_DIR}/scy_based.lib)
set(LIBSOURCEY_SCY_CRYPTO_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_crypto.lib debug ${LIBSOURCEY_LIB_DIR}/scy_cryptod.lib)
set(LIBSOURCEY_SCY_HTTP_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_http.lib debug ${LIBSOURCEY_LIB_DIR}/scy_httpd.lib)
set(LIBSOURCEY_SCY_JSON_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_json.lib debug ${LIBSOURCEY_LIB_DIR}/scy_jsond.lib)
set(LIBSOURCEY_SCY_NET_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_net.lib debug ${LIBSOURCEY_LIB_DIR}/scy_netd.lib)
set(LIBSOURCEY_SCY_SCHED_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_sched.lib debug ${LIBSOURCEY_LIB_DIR}/scy_schedd.lib)
set(LIBSOURCEY_SCY_SOCKETIO_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_socketio.lib debug ${LIBSOURCEY_LIB_DIR}/scy_socketiod.lib)
set(LIBSOURCEY_SCY_STUN_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_stun.lib debug ${LIBSOURCEY_LIB_DIR}/scy_stund.lib)
set(LIBSOURCEY_SCY_SYMPLE_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_symple.lib debug ${LIBSOURCEY_LIB_DIR}/scy_sympled.lib)
set(LIBSOURCEY_SCY_TURN_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_turn.lib debug ${LIBSOURCEY_LIB_DIR}/scy_turnd.lib)
set(LIBSOURCEY_SCY_UTIL_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_util.lib debug ${LIBSOURCEY_LIB_DIR}/scy_utild.lib)
set(LIBSOURCEY_SCY_WEBRTC_LIBS optimized ${LIBSOURCEY_LIB_DIR}/scy_webrtc.lib debug ${LIBSOURCEY_LIB_DIR}/scy_webrtcd.lib)

#set(LIBSOURCEY_SCY_ARCHO_LIBS ${LIBSOURCEY_LIB_DIR}/scy_archod.lib)
#set(LIBSOURCEY_SCY_AV_LIBS ${LIBSOURCEY_LIB_DIR}/scy_avd.lib)
#set(LIBSOURCEY_SCY_BASE_LIBS ${LIBSOURCEY_LIB_DIR}/scy_based.lib)
#set(LIBSOURCEY_SCY_CRYPTO_LIBS ${LIBSOURCEY_LIB_DIR}/scy_cryptod.lib)
#set(LIBSOURCEY_SCY_HTTP_LIBS ${LIBSOURCEY_LIB_DIR}/scy_httpd.lib)
#set(LIBSOURCEY_SCY_JSON_LIBS ${LIBSOURCEY_LIB_DIR}/scy_jsond.lib)
#set(LIBSOURCEY_SCY_NET_LIBS ${LIBSOURCEY_LIB_DIR}/scy_netd.lib)
#set(LIBSOURCEY_SCY_SCHED_LIBS ${LIBSOURCEY_LIB_DIR}/scy_schedd.lib)
#set(LIBSOURCEY_SCY_SOCKETIO_LIBS ${LIBSOURCEY_LIB_DIR}/scy_socketiod.lib)
#set(LIBSOURCEY_SCY_STUN_LIBS ${LIBSOURCEY_LIB_DIR}/scy_stund.lib)
#set(LIBSOURCEY_SCY_SYMPLE_LIBS ${LIBSOURCEY_LIB_DIR}/scy_sympled.lib)
#set(LIBSOURCEY_SCY_TURN_LIBS ${LIBSOURCEY_LIB_DIR}/scy_turnd.lib)
#set(LIBSOURCEY_SCY_UTIL_LIBS ${LIBSOURCEY_LIB_DIR}/scy_utild.lib)
#set(LIBSOURCEY_SCY_WEBRTC_LIBS ${LIBSOURCEY_LIB_DIR}/scy_webrtcd.lib)

#
# webRTC Library dependency
#
set(WEBRTC_WIN_API_DIR ${CMAKE_SOURCE_DIR}/External/Libs/webrtc-16937-win-x64/include CACHE PATH "")

set(WEBRTC_WIN_API_LIBS optimized ${CMAKE_SOURCE_DIR}/External/Libs/webrtc-16937-win-x64/lib/x64/Release/libwebrtc_full.lib debug ${CMAKE_SOURCE_DIR}/External/Libs/webrtc-16937-win-x64/lib/x64/Debug/libwebrtc_full.lib)
set(LIBSOURCEY_LIBUV_LIBS optimized ${LIBSOURCEY_API_DIR}/vendor/lib/libuv.lib debug ${LIBSOURCEY_API_DIR}/vendor/lib/libuvd.lib)
set(LIBSOURCEY_HTTP_PARSER_LIBS optimized ${LIBSOURCEY_API_DIR}/vendor/lib/http_parser.lib debug ${LIBSOURCEY_API_DIR}/vendor/lib/http_parserd.lib)
set(LIBSOURCEY_MINIZIP_LIBS optimized ${LIBSOURCEY_API_DIR}/vendor/lib/minizip.lib debug ${LIBSOURCEY_API_DIR}/vendor/lib/minizipd.lib)
set(LIBSOURCEY_ZLIB_LIBS optimized ${LIBSOURCEY_API_DIR}/vendor/lib/zlib.lib debug ${LIBSOURCEY_API_DIR}/vendor/lib/zlibd.lib)
set(LIBSOURCEY_ZLIB_STATIC_LIBS optimized ${LIBSOURCEY_API_DIR}/vendor/lib/zlibstatic.lib debug ${LIBSOURCEY_API_DIR}/vendor/lib/zlibstaticd.lib)

#set(WEBRTC_WIN_API_LIBS ${CMAKE_SOURCE_DIR}/External/Libs/webrtc-16937-win-x64/lib/x64/Debug/libwebrtc_full.lib)
#set(LIBSOURCEY_LIBUV_LIBS ${LIBSOURCEY_API_DIR}/vendor/lib/libuvd.lib)
#set(LIBSOURCEY_HTTP_PARSER_LIBS ${LIBSOURCEY_API_DIR}/vendor/lib/http_parserd.lib)
#set(LIBSOURCEY_MINIZIP_LIBS ${LIBSOURCEY_API_DIR}/vendor/lib/minizipd.lib)
#set(LIBSOURCEY_ZLIB_LIBS ${LIBSOURCEY_API_DIR}/vendor/lib/zlibd.lib)
#set(LIBSOURCEY_ZLIB_STATIC_LIBS ${LIBSOURCEY_API_DIR}/vendor/lib/zlibstaticd.lib)

set(LIBSOURCEY_WINDOWS_LIBS ws2_32.lib psapi.lib iphlpapi.lib userenv.lib dsound.lib strmiids.lib Secur32.lib msdmo.lib dmoguids.lib wmcodecdspuuid.lib winmm.lib)

#
# Spinnaker
#
find_path(SPINNAKER_INC_DIR spinnaker.h)
find_path(SPINNAKER_LIB_DIR Spinnaker_v140.lib)
set(SPINNAKER_LIBS optimized ${SPINNAKER_LIB_DIR}/Spinnaker_v140.lib debug ${SPINNAKER_LIB_DIR}/Spinnakerd_v140.lib)
#
# More Qt
#
#get_target_property(QtCore_location Qt5::Core LOCATION)

#
# Disable some warnings in Windows
#
if(MSVC)
	add_definitions(/DWIN32)
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} /MDd CACHE STRING "" FORCE)
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} /MD CACHE STRING "" FORCE)
	# if(CMAKE_CONFIGURATION_TYPES,Debug)
		# add_definitions(/MDd)
	# else(CMAKE_CONFIGURATION_TYPES,Debug)
		# add_definitions(/MD)
	# endif()
    add_definitions(/W1)
	add_definitions(/D_SCL_SECURE_NO_WARNINGS)
	add_definitions(/D__STDC_LIMIT_MACROS)
	add_definitions(/wd4251)
	add_definitions(/wd4244 /wd4800 /wd4146 /wd4355 /wd4530 /wd4819)
	add_definitions(-DUSE_SHADER_CODE)
	add_definitions(-DUSE_GPU_COLORCVT)
	if(OCULUS_1_8_0)
		add_definitions(-DOCULUS_1_8_0)
	endif()
#	set(CMAKE_EXE_LINKER_FLAGS
#      "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:msvcrt.lib")
	  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
	  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /MDd /Od" CACHE STRING "Flags used by the C++ Debug" FORCE)
	set(CMAKE_CXX_FLAGS_RELEASE "/MD" CACHE STRING "Flags used by C++ Release" FORCE)
endif()

#
# Arrange project files in folders/filters
#
set_property(GLOBAL PROPERTY USE_FOLDERS On) 

# The Qt5Widgets_INCLUDES also includes the include directories for
# dependencies QtCore and QtGui

include_directories(
	${Qt5Widgets_INCLUDES}
	${Qt5Gui_PRIVATE_INCLUDE_DIRS}
#	${QT5P7_INC_DIR}
#	${QT5P7_INC_GUI_DIR}
#	${QT5P7_INC_CORE_DIR}
#	${QT5P7_INC_QML_DIR}
#	${QT5P7_INC_WIDGETS_DIR}
#	${QT5P7_INC_QUICK_DIR}
	${FFMPEG_INC_DIR}
	${LIBSOURCEY_INC_DIR}
	${LIBZIP_INC_DIR}
	${GLEW_INC_DIR}
	${OpenCV_INC_DIR}
	${OVR_INC_DIR}
	${OVRKERNEL_INC_DIR}
	${EIGEN_INC_DIR}
	${SPINNAKER_INC_DIR}
   	./ 
	./Include/
	./Capture/
	./Process/
	./GPU/
	./GPU/Graphics/
	./GPU/CUDA/
	./GPU/CUDAKernels/
	./View/
	./Common/
	./Cloud/
	${LIBSOURCEY_API_DIR}/include
	${LIBSOURCEY_API_DIR}/vendor/include
	${WEBRTC_WIN_API_DIR}
	${WEBRTC_WIN_API_DIR}/third_party/boringssl/src/include
)

# We need add -DQT_WIDGETS_LIB when using QtWidgets in Qt 5.
add_definitions(${Qt5Widgets_DEFINITIONS} -DENABLE_LOG)

#
# Executables fail to build with Qt 5 in the default configuration
#
set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

# This line includes the current Dir of CMake. The xxx.ui file is
# converted to a ui_xxx.h file by uic.exe. The header ui_xxx.h file
# is located in the build directory. To make this header visible to
# MSVS the build path needs to be included.
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(MAIN_FILES
	./main.cpp
)
set(CAPTURE_H_FILES
	./Capture/AudioThread.h
	./Capture/Capture.h
	./Capture/CameraView.h
	./Capture/PlaybackView.h
	./Capture/CaptureAudio.h
	./Capture/CaptureDevices.h
	./Capture/CaptureDShow.h
	./Capture/CaptureImageFile.h
	./Capture/CaptureLiDAR.h
	./Capture/CaptureThread.h
	./Capture/PlaybackThread.h
	./Capture/VideoCapture.h
	./Capture/BannerThread.h
	./Capture/config.h
	./Capture/LIDARThread.h
	./Capture/signaler.h
	./Capture/multiplexmediacapturer.h
	./Capture/captureptgrey.h
	
)
set(CAPTURE_CXX_FILES	
	./Capture/AudioThread.cpp
	./Capture/CameraView.cpp
	./Capture/PlaybackView.cpp
	./Capture/Capture.cpp
	./Capture/CaptureAudio.cpp
	./Capture/CaptureDevices.cpp
	./Capture/CaptureDShow.cpp
	./Capture/CaptureImageFile.cpp
	./Capture/CaptureLiDAR.cpp
	./Capture/CaptureThread.cpp
	./Capture/PlaybackThread.cpp
	./Capture/VideoCapture.cpp
	./Capture/BannerThread.cpp
	./Capture/signaler.cpp
	./Capture/LIDARThread.cpp
	./Capture/multiplexmediacapturer.cpp
	./Capture/CapturePtGrey.cpp
)
set(PROCESS_H_FILES
	./Process/D360Parser.h
	./Process/D360Process.h
	./Process/D360Stitcher.h
	./Process/PlaybackStitcher.h
	./Process/OculusViewer.h
	./Process/PtsParser.h
	./Process/StreamProcess.h
	./Process/CalibProcess.h
	./Process/SnapshotDetector.h
	./Process/SelfCalibrator.h
	./Process/CoreEngine.h
	)
set(PROCESS_CXX_FILES	
	./Process/D360Parser.cpp
	./Process/D360Process.cpp
	./Process/D360Stitcher.cpp
	./Process/PlaybackStitcher.cpp
	./Process/OculusViewer.cpp
	./Process/PtsParser.cpp
	./Process/StreamProcess.cpp
	./Process/CalibProcess.cpp
	./Process/SnapshotDetector.cpp
	./Process/SelfCalibrator.cpp
	./Process/CoreEngine.cpp
)

set(GPU_H_FILES
	./GPU/3DMath.h
	./GPU/SingleViewProcess.h
	./GPU/SinglePanoramaUnit.h
	./GPU/GPUProgram.h
	./GPU/GPUColorCvt.h
	./GPU/GPUGainCompensation.h
	./GPU/GPUWeightMap.h
	./GPU/GPUUnwarp.h
	./GPU/GPUPanoramaWeight.h
	./GPU/GPUSeam.h
	./GPU/GPUBoundary.h
	./GPU/GPUFeathering.h
	./GPU/GPUWeightVisualizer.h
	./GPU/GPUComposite.h
	./GPU/GPUBanner.h
	./GPU/GPUPanoramaPostProcessing.h
	./GPU/GPUFinalPanorama.h
	./GPU/GPUNodalBlending.h
)
set(GPU_CXX_FILES
	./GPU/3DMath.cpp
	./GPU/SingleViewProcess.cpp
	./GPU/SinglePanoramaUnit.cpp
	./GPU/GPUProgram.cpp
	./GPU/GPUColorCvt.cpp
	./GPU/GPUGainCompensation.cpp
	./GPU/GPUWeightMap.cpp
	./GPU/GPUUnwarp.cpp
	./GPU/GPUPanoramaWeight.cpp
	./GPU/GPUSeam.cpp
	./GPU/GPUBoundary.cpp
	./GPU/GPUFeathering.cpp
	./GPU/GPUWeightVisualizer.cpp
	./GPU/GPUComposite.cpp
	./GPU/GPUBanner.cpp
	./GPU/GPUPanoramaPostProcessing.cpp
	./GPU/GPUFinalPanorama.cpp
	./GPU/GPUNodalBlending.cpp
)

set(GRAPHICS_H_FILES
	./GPU/Graphics/GLSLColorCvt.h
	./GPU/Graphics/GLSLGainCompensation.h
	./GPU/Graphics/GLSLUnwarp.h
	./GPU/Graphics/GLSLWeightMap.h
	./GPU/Graphics/GLSLFeathering.h
	./GPU/Graphics/GLSLMultibandBlending.h
	./GPU/Graphics/GLSLWeightVisualizer.h
	./GPU/Graphics/GLSLNodalBlending.h
	./GPU/Graphics/GLSLBoundary.h
	./GPU/Graphics/GLSLGaussianBlur.h
	./GPU/Graphics/GLSLBoxBlur.h
	./GPU/Graphics/GLSLLaplacian.h
	./GPU/Graphics/GLSLBlending.h
	./GPU/Graphics/GLSLResize.h
	./GPU/Graphics/GLSLPyramidSum.h
	./GPU/Graphics/GLSLFinalPanorama.h
	./GPU/Graphics/GLSLPanoramaInteract.h
	./GPU/Graphics/GLSLPanoramaPostProcessing.h
	./GPU/Graphics/GLSLBanner.h
)

set(GRAPHICS_CXX_FILES
	./GPU/Graphics/GLSLColorCvt.cpp
	./GPU/Graphics/GLSLGainCompensation.cpp
	./GPU/Graphics/GLSLUnwarp.cpp
	./GPU/Graphics/GLSLWeightMap.cpp
	./GPU/Graphics/GLSLFeathering.cpp
	./GPU/Graphics/GLSLMultibandBlending.cpp
	./GPU/Graphics/GLSLWeightVisualizer.cpp
	./GPU/Graphics/GLSLNodalBlending.cpp
	./GPU/Graphics/GLSLBoundary.cpp
	./GPU/Graphics/GLSLGaussianBlur.cpp
	./GPU/Graphics/GLSLBoxBlur.cpp
	./GPU/Graphics/GLSLLaplacian.cpp
	./GPU/Graphics/GLSLBlending.cpp
	./GPU/Graphics/GLSLResize.cpp
	./GPU/Graphics/GLSLPyramidSum.cpp
	./GPU/Graphics/GLSLFinalPanorama.cpp
	./GPU/Graphics/GLSLPanoramaInteract.cpp
	./GPU/Graphics/GLSLPanoramaPostProcessing.cpp
	./GPU/Graphics/GLSLBanner.cpp
)
set(CUDA_H_FILES
	./GPU/CUDA/CUDAColorCvt.h
	./GPU/CUDA/CUDAGainCompensation.h
	./GPU/CUDA/CUDAWeightMap.h
	./GPU/CUDA/CUDAUnwarp.h
	./GPU/CUDA/CUDABoundary.h
	./GPU/CUDA/CUDAFeathering.h
	./GPU/CUDA/CUDAWeightVisualizer.h
	./GPU/CUDA/CUDABanner.h
	./GPU/CUDA/CUDAPanoramaPostProcessing.h
	./GPU/CUDA/CUDAFinalPanorama.h
	./GPU/CUDA/CUDANodalBlending.h
)

set(CUDA_CXX_FILES
	./GPU/CUDA/CUDAColorCvt.cpp
	./GPU/CUDA/CUDAGainCompensation.cpp
	./GPU/CUDA/CUDAWeightMap.cpp
	./GPU/CUDA/CUDAUnwarp.cpp
	./GPU/CUDA/CUDABoundary.cpp
	./GPU/CUDA/CUDAFeathering.cpp
	./GPU/CUDA/CUDAWeightVisualizer.cpp
	./GPU/CUDA/CUDABanner.cpp
	./GPU/CUDA/CUDAPanoramaPostProcessing.cpp
	./GPU/CUDA/CUDAFinalPanorama.cpp
	./GPU/CUDA/CUDANodalBlending.cpp
)

set(CUDA_KERNEL_CXX_FILES
	./GPU/CUDAKernels/CudaCommon.cuh
	./GPU/CUDAKernels/Kernels_ColorCvt.cu
	./GPU/CUDAKernels/Kernels_GainCompensation.cu
	./GPU/CUDAKernels/Kernels_WeightMap.cu
	./GPU/CUDAKernels/Kernels_Unwarp.cu
	./GPU/CUDAKernels/Kernels_Seam.cu
	./GPU/CUDAKernels/Kernels_Boundary.cu
	./GPU/CUDAKernels/Kernels_Feathering.cu
	./GPU/CUDAKernels/Kernels_WeightVisualizer.cu
	./GPU/CUDAKernels/Kernels_Banner.cu
	./GPU/CUDAKernels/Kernels_PanoramaPostProcessing.cu
	./GPU/CUDAKernels/Kernels_FinalPanorama.cu
	./GPU/CUDAKernels/Kernels_NodalBlending.cu
)

set(VIEW_H_FILES
	./View/MCQmlCameraView.h
	./View/QmlInteractiveView.h
	./View/QmlMainWindow.h
	./View/QmlRecentDialog.h
	./View/QmlTakeManagement.h
	./View/QmlApplicationSetting.h
	./View/ConsoleRunner.h
	./view/Pixmap.h
)
set(VIEW_CXX_FILES	
	./View/MCQmlCameraView.cpp
	./View/QmlInteractiveView.cpp
	./View/QmlMainWindow.cpp
	./View/QmlRecentDialog.cpp
	./View/QmlTakeManagement.cpp
	./View/QmlApplicationSetting.cpp
	./View/ConsoleRunner.cpp
	./view/Pixmap.cpp
)
set(COMMON_H_FILES
	./Common/BaseFfmpeg.hpp
	./Common/common.h
	./Common/FfmpegUtils.hpp
#	./Common/GlobalAnimSettings.h
	./Common/PanoQueue.h
	./Common/QTHandler.h
	./Common/SharedImageBuffer.h
	./Common/StreamFfmpeg.hpp
	./Common/TLogger.h
	./Common/TSystemTypes.h
	./Common/PanoLog.h
	./Common/SlotInfo.h
	./Common/TakeMgrTreeModel.h
	./Common/ConfigZip.h
	./External/Libs/libzip/include/zip.h
	./External/Libs/libzip/include/zipconf.h
#	./Structures.h
	./Include/Config.h
	./Include/define.h
)

set(COMMON_CXX_FILES	
	./Common/BaseFfmpeg.cpp
	./Common/common.cpp
#	./Common/GlobalAnimSettings.cpp
	./Common/PanoQueue.cpp
	./Common/QTHandler.cpp
	./Common/SharedImageBuffer.cpp
	./Common/StreamFfmpeg.cpp
	./Common/TLogger.cpp
	./Common/TSystemSpec.cpp
	./Common/PanoLog.cpp
	./Common/SlotInfo.cpp
	./Common/TakeMgrTreeModel.cpp
	./Common/ConfigZip.cpp
)


set(CLOUD_H_FILES
	./Cloud/RosDataConverter.h
	./Cloud/rosDuration.h
	./Cloud/rosHeader.h
	./Cloud/rosMsg.h
	./Cloud/PointCloud2.h
	./Cloud/PointField.h
	./Cloud/rosTime.h
)

set(CLOUD_CXX_FILES
	./Cloud/RosDataConverter.cpp
	./Cloud/rosDuration.cpp
	./Cloud/rosTime.cpp
)

#
# QML files
#

set(QML_FILES 
	main.qml
	qml/Header.qml
	qml/Toolbox.qml
	qml/Recent.qml
	qml/RecentGriditem.qml	
	qml/RecentListitem.qml
	qml/TemplateGriditem.qml	
	qml/Status.qml
	qml/PlayStatus.qml
)
set(QML_FILES_COMPONENTS
	qml/components/Aboutbox.qml
	qml/components/Anaglyphbox.qml
	qml/components/Exposurebox.qml
	qml/components/Helpbox.qml
	qml/components/Interactivebox.qml
	qml/components/InteractiveView.qml
	qml/components/Livebox.qml
	qml/components/Savebox.qml
	qml/components/Snapshotbox.qml
	qml/components/Sphericalbox.qml
	qml/components/Topbottombox.qml
	qml/components/LiveSettingbox.qml
	qml/components/DshowSettingbox.qml
	qml/components/DeckinkSettingbox.qml
	qml/components/LiveAudioitem.qml
	qml/components/LiveCamListitem.qml	
	qml/components/VideoSettingbox.qml
	qml/components/VideoListitem.qml
	qml/components/VideoEmptyitem.qml
	qml/components/Streaming.qml
	qml/components/ImageSettingbox.qml
	qml/components/ImageListitem.qml
	qml/components/Spliter.qml
	qml/components/FloatingWindow.qml
	qml/components/FloatingStitchWindow.qml
	qml/components/MCCameraView.qml
	qml/components/ToolbarItem.qml
	qml/components/CameraParameterItem.qml
	qml/components/TipMessage.qml
	qml/components/MCStitchCameraView.qml
	qml/components/SphericalView.qml
	qml/components/LiveView.qml
	qml/components/MCVideoWindow.qml
	qml/components/MCCalibrationCameraView.qml
	qml/components/MCStitchVideoWindow.qml
	qml/components/MdiArea.qml
	qml/components/ToolWindowPositions.js
	qml/components/LiveCamViewPosition.js
	qml/components/ExposureDialog.qml
	qml/components/ToolWindow.qml
	qml/components/FloatingInteractiveWindow.qml
	qml/components/MCInteractCameraView.qml
	qml/components/MCInteractVideoWindow.qml
	qml/components/CameraPreview.qml
	qml/components/CameraSettingbox.qml
	qml/components/VideoGlobalSettings.qml
	qml/components/LiveTempCamSetting.qml
	qml/components/ImageTempCamSetting.qml
	qml/components/GroupCtrl.qml
	qml/components/ControlPoint.qml
	qml/components/SeamLabel.qml
	qml/components/Notification.qml
	qml/components/NotificationHistory.qml
	qml/components/NotificationItem.qml
	qml/components/ExposureCtrl.qml
	qml/components/Line.qml
	qml/components/BannerCtrl.qml
	qml/components/CameraParams.qml
	qml/components/OculusCtrl.qml
	qml/components/OculusSwitch.qml
	qml/components/BlendSettings.qml
	qml/components/ScreenNumber.qml
	qml/components/WeightmapWindow.qml
	qml/components/WeightmapSettingbox.qml
	qml/components/ApplicationSetting.qml
	qml/components/CameraParamsSettingbox.qml
	qml/components/CalibrationSettingbox.qml
	qml/components/CameraTemplateSettingbox.qml
	qml/components/StitchCameraTemplateSettingbox.qml	
	qml/components/RigSaveSettingbox.qml	
	qml/components/RigLoadSettingbox.qml	
	qml/components/TemplateLoadSettingbox.qml	
	qml/components/RigSaveCameraListItem.qml
	qml/components/RigLoadCameraListItem.qml	
	qml/components/TemplateCameraListItem.qml		
	qml/components/WeightmapTopCtrl.qml
	qml/components/TakeManagementSettingbox.qml
	qml/components/CommentDialog.qml
	qml/components/NodalListitem.qml	
	qml/components/CTSettingbox.qml
	qml/components/MCCheckBox.qml
	qml/components/DisplayLUTControl.qml
	qml/components/HiResItem.qml
	qml/components/L3DButton.qml
)

set(QML_FILES_COMPONENTS_CHART
	qml/components/QMLChartJs/Chart.js
	qml/components/QMLChartJs/main.qml
	qml/components/QMLChartJs/QChartJs.qml
	qml/components/QMLChartJs/QChartJsTypes.js
	qml/components/QMLChartJs/QMLChartData.js
)

set(QML_FILES_CONTROLS
	qml/controls/FlatCombo.qml
	qml/controls/FlatText.qml
	qml/controls/FlatSpin.qml
)

source_group( "Source Files" FILES ${MAIN_FILES})

source_group( "Header Files\\Capture" FILES ${CAPTURE_H_FILES} )
source_group( "Source Files\\Capture" FILES ${CAPTURE_CXX_FILES} )

source_group( "Header Files\\Process" FILES ${PROCESS_H_FILES} )
source_group( "Source Files\\Process" FILES ${PROCESS_CXX_FILES} )

source_group( "Header Files\\GPU" FILES ${GPU_H_FILES} )
source_group( "Source Files\\GPU" FILES ${GPU_CXX_FILES} )

source_group( "Header Files\\GPU\\Graphics" FILES ${GRAPHICS_H_FILES} )
source_group( "Source Files\\GPU\\Graphics" FILES ${GRAPHICS_CXX_FILES} )

source_group( "Header Files\\GPU\\CUDA" FILES ${CUDA_H_FILES} )
source_group( "Source Files\\GPU\\CUDA" FILES ${CUDA_CXX_FILES} )


source_group( "Source Files\\GPU\\CUDAKernels" FILES ${CUDA_KERNEL_CXX_FILES} )

source_group( "Header Files\\View" FILES ${VIEW_H_FILES} )
source_group( "Source Files\\View" FILES ${VIEW_CXX_FILES} )

source_group( "Header Files\\Common" FILES ${COMMON_H_FILES} )
source_group( "Source Files\\Common" FILES ${COMMON_CXX_FILES} )

source_group( "Header Files\\Cloud" FILES ${CLOUD_H_FILES} )
source_group( "Source Files\\Cloud" FILES ${CLOUD_CXX_FILES} )

source_group( "QML Files" FILES ${QML_FILES} ${QML_FILES_COMPONENTS} ${QML_FILES_CONTROLS} )
source_group( "QML Files\\components\\QMLChartJs" FILES ${QML_FILES_COMPONENTS_CHART})

#
# This line is necessary to cal uic.exe and creates the ui_xxx.h
# files from the xxx.ui files. The header is created the first time
# the project is compiled
#
#
# Add Qt Resources
#
#qt5_add_resources(UI_RESOURCES_RCC ${CMAKE_SOURCE_DIR}/UI/PanoOne.qrc)
#source_group("Resource Files" FILES ${CMAKE_SOURCE_DIR}/UI/PanoOne.qrc)

qt5_add_resources(UI_RESOURCES_RCC ${CMAKE_SOURCE_DIR}/qml.qrc)
source_group("Resource Files" FILES ${CMAKE_SOURCE_DIR}/qml.qrc)
if(CUDA_FOUND)
cuda_add_executable(Look3D 
	${MAIN_FILES} 
	${CAPTURE_H_FILES} ${CAPTURE_CXX_FILES} 
	${PROCESS_H_FILES} ${PROCESS_CXX_FILES} 
	${GPU_H_FILES} ${GPU_CXX_FILES}
	${GRAPHICS_H_FILES} ${GRAPHICS_CXX_FILES} 
	${CUDA_H_FILES} ${CUDA_CXX_FILES} 
	${CUDA_KERNEL_CXX_FILES}
	${VIEW_H_FILES} ${VIEW_CXX_FILES} 
	${COMMON_H_FILES} ${COMMON_CXX_FILES}
	${CLOUD_H_FILES} ${CLOUD_CXX_FILES}
	${QML_FILES} ${QML_FILES_COMPONENTS} ${QML_FILES_COMPONENTS_CHART} ${QML_FILES_CONTROLS} 
	${FORMS_HEADERS} ${HEADERS_MOC} ${UI_RESOURCES_RCC}
)
else()
add_executable(Look3D 
	${MAIN_FILES} 
	${CAPTURE_H_FILES} ${CAPTURE_CXX_FILES} 
	${PROCESS_H_FILES} ${PROCESS_CXX_FILES}
	${GPU_H_FILES} ${GPU_CXX_FILES}
	${GRAPHICS_H_FILES} ${GRAPHICS_CXX_FILES} 
	${VIEW_H_FILES} ${VIEW_CXX_FILES} 
	${COMMON_H_FILES} ${COMMON_CXX_FILES}
	${CLOUD_H_FILES} ${CLOUD_CXX_FILES}
	${QML_FILES} ${QML_FILES_COMPONENTS} ${QML_FILES_COMPONENTS_CHART} ${QML_FILES_CONTROLS} 
	${FORMS_HEADERS} ${HEADERS_MOC} ${UI_RESOURCES_RCC}
)
endif()
				

#set_target_properties(Look3D
#				PROPERTIES LINK_FLAGS "/level='requireAdministrator' /uiAccess='false' /ignore:4819")
set_target_properties(Look3D
				PROPERTIES LINK_FLAGS "/ignore:4819")

target_link_libraries(${PROJECT_NAME} 
	Qt5::Widgets Qt5::OpenGL Qt5::Qml Qt5::Quick Qt5::Xml
#	Qt5::Network 
#	${QT5P7_LIB_OpenGL_LIBS}
#	${QT5P7_LIB_Core_LIBS}
#	${QT5P7_LIB_Quick_LIBS}
#	${QT5P7_LIB_Xml_LIBS}
#	${QT5P7_LIB_Qml_LIBS}
#	${QT5P7_LIB_Widgets_LIBS}
#	${QT5P7_LIB_gui_LIBS}
#	${QT5P7_LIB_Network_LIBS}
	${OpenCV_LIBS}
	${OPENGL_LIBRARIES}
	${FFMPEG_AVCODEC_LIBS}
	${FFMPEG_AVFORMAT_LIBS}
	${FFMPEG_AVUTIL_LIBS}
	${FFMPEG_AVDEVICE_LIBS}
	${FFMPEG_SWSCALE_LIBS}
	${FFMPEG_SWRESAMPLE_LIBS}
	${LIBZIP_ZIP_LIBS}
	${GLEW_LIBS}
	${LIBSOURCEY_SCY_ARCHO_LIBS}
	${LIBSOURCEY_SCY_AV_LIBS}
	${LIBSOURCEY_SCY_BASE_LIBS}
	${LIBSOURCEY_SCY_CRYPTO_LIBS}
	${LIBSOURCEY_SCY_HTTP_LIBS}
	${LIBSOURCEY_SCY_JSON_LIBS}
	${LIBSOURCEY_SCY_NET_LIBS}
	${LIBSOURCEY_SCY_SCHED_LIBS}
	${LIBSOURCEY_SCY_SOCKETIO_LIBS}
	${LIBSOURCEY_SCY_STUN_LIBS}
	${LIBSOURCEY_SCY_SYMPLE_LIBS}
	${LIBSOURCEY_SCY_TURN_LIBS}
	${LIBSOURCEY_SCY_UTIL_LIBS}
	${LIBSOURCEY_SCY_WEBRTC_LIBS}
	${WEBRTC_WIN_API_LIBS}
	${LIBSOURCEY_LIBUV_LIBS}
	${LIBSOURCEY_HTTP_PARSER_LIBS}
	${LIBSOURCEY_MINIZIP_LIBS}
	${LIBSOURCEY_ZLIB_LIBS}
	${LIBSOURCEY_ZLIB_STATIC_LIBS}
	${LIBSOURCEY_WINDOWS_LIBS}
	${OVR_LIBS}
	${WIN_VFW}
	${SPINNAKER_LIBS}
	)