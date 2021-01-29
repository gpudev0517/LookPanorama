/************************************************************************/
/* qt-opencv-multithreaded:                                             */
/* A multithreaded OpenCV application using the Qt framework.           */
/*                                                                      */
/* Structures.h                                                         */
/*                                                                      */
/* Nick D'Ademo <nickdademo@gmail.com>                                  */
/*                                                                      */
/* Copyright (c) 2012-2013 Nick D'Ademo                                 */
/*                                                                      */
/* Permission is hereby granted, free of charge, to any person          */
/* obtaining a copy of this software and associated documentation       */
/* files (the "Software"), to deal in the Software without restriction, */
/* including without limitation the rights to use, copy, modify, merge, */
/* publish, distribute, sublicense, and/or sell copies of the Software, */
/* and to permit persons to whom the Software is furnished to do so,    */
/* subject to the following conditions:                                 */
/*                                                                      */
/* The above copyright notice and this permission notice shall be       */
/* included in all copies or substantial portions of the Software.      */
/*                                                                      */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,      */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF   */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                */
/* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS  */
/* BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN   */
/* ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN    */
/* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     */
/* SOFTWARE.                                                            */
/*                                                                      */
/************************************************************************/

#ifndef STRUCTURES_H
#define STRUCTURES_H

// Qt
#include <QtCore/QRect>
#include <QTime>
#include <qlocale.h>
#include "3DMath.h"
#include "Capture.h"

struct ImageProcessingSettings{
    int smoothType;
    int smoothParam1;
    int smoothParam2;
    double smoothParam3;
    double smoothParam4;
    int dilateNumberOfIterations;
    int erodeNumberOfIterations;
    int flipCode;
    double cannyThreshold1;
    double cannyThreshold2;
    int cannyApertureSize;
    bool cannyL2gradient;
};

struct ImageProcessingFlags{
    bool grayscaleOn;
    bool smoothOn;
    bool dilateOn;
    bool erodeOn;
    bool flipOn;
    bool cannyOn;
};

struct MouseData{
    QRect selectionBox;
    bool leftButtonRelease;
    bool rightButtonRelease;
};

struct ThreadStatisticsData{
    float averageFPS;
    int nFramesProcessed;
	int nAudioFrames;
	int instantFPS;
	int elapesedTime;

	// lastFramesProcessed: is for Calculating the duration of Take
	QString toString(int fps, int lastFramesProcessed)
	{
		int frames = nFramesProcessed - 1;
		frames -= lastFramesProcessed;
		if (frames < 0) frames = 0;
		int seconds = frames / fps;
		int minutes = seconds / 60;
		int hours = minutes / 60;
		
		frames = frames % fps;
		seconds = seconds % 60;
		minutes = minutes % 60;
		
		QTime time(hours, minutes, seconds);
		QLocale conv;
		QString strFF = conv.toString(frames);
		switch (strFF.length())
		{
		case 1:
			strFF = "0" + strFF;
			break;
		case 2:
			strFF = strFF;
			break;
		default:
			break;
		}
		QString strElapsedTime = time.toString() + "." + strFF;
		return strElapsedTime;
	}
};

struct CameraParameters
{
	enum LensType
	{
		LensType_ptLens_Standard,
		LensType_ptLens_Fullframe_Fisheye,
		LensType_ptLens_Circular_Fisheye,
		LensType_opencvLens_Standard,
		LensType_opencvLens_Fisheye,
	};

	int   m_cameraNumber;
	LensType   m_lensType;
	float m_yaw;
	float m_pitch;
	float m_roll;
	float m_fov;
	float m_fovy;
	float m_k1;
	float m_k2;
	float m_k3;
	float m_offset_x;
	float m_offset_y;
	float m_expOffset;
	float m_xrad1; // feathering crop (0.0:none~1.0:all)
	float m_xrad2;
	float m_yrad1;
	float m_yrad2;

	/*float m_focal35mm; // not in use, since it's initial focal estimate
	float m_fisheyeCoffX; 
	float m_fisheyeCoffY;
	float m_yaw;
	float m_pitch;
	float m_roll;
	float m_f;
	float m_k1;
	float m_k2;
	float m_k3;
	float m_offset_x;
	float m_offset_y;
	float m_fisheye_radius;
	float m_ccdwidthinMM;
	float m_ccdheightinMM;*/

	CameraParameters()
	{
		//m_ccdwidthinMM  = 11.27; // Based on Ximea xiQ USB Vision Camera - other cameras might vary
		//m_ccdheightinMM = 11.27;
		m_lensType = LensType_ptLens_Fullframe_Fisheye;		// Default is fisheye lens
		m_fov = 240;
		m_expOffset = 0.0f;
		m_xrad1 = m_xrad2 = 1.0f;
		m_yrad1 = m_yrad2 = 1.0f;
		m_yaw = 0.0f;
		m_pitch = 0.0f;
		m_roll = 0.0f;
		m_k1 = m_k2 = m_k3 = 0;
		m_offset_x = m_offset_y = 0;
	}

	CameraParameters& operator = (const CameraParameters& other)
	{
		m_cameraNumber = other.m_cameraNumber;
		m_lensType = other.m_lensType;
		m_yaw = other.m_yaw;
		m_pitch = other.m_pitch;
		m_roll = other.m_roll;
		m_fov = other.m_fov;
		m_fovy = other.m_fovy;
		m_k1 = other.m_k1;
		m_k2 = other.m_k2;
		m_k3 = other.m_k3;
		m_offset_x = other.m_offset_x;
		m_offset_y = other.m_offset_y;
		m_expOffset = other.m_expOffset;
		m_xrad1 = other.m_xrad1;
		m_xrad2 = other.m_xrad2;
		m_yrad1 = other.m_yrad1;
		m_yrad2 = other.m_yrad2;

		return *this;
	}

	bool isFisheye()
	{
		return m_lensType == LensType_ptLens_Circular_Fisheye ||
			m_lensType == LensType_ptLens_Fullframe_Fisheye ||
			m_lensType == LensType_opencvLens_Fisheye;
	}
};


struct CameraInput
{
	CameraInput()
	{
		clear();
	}

	CameraInput& operator = (const CameraInput& other)
	{		
		name = other.name;
		stereoType = other.stereoType;
		audioName = other.audioName;
		audioType = other.audioType;
		fileDir = other.fileDir;
		filePrefix = other.filePrefix;
		fileExt = other.fileExt;
		exposure = other.exposure;
		xres = other.xres;
		yres = other.yres;
		fps = other.fps;
		playbackfps = other.playbackfps;
		valid = other.valid;
		cameraType = other.cameraType;
		m_cameraParams = other.m_cameraParams;

		return *this;
	}

	void clear()
	{
		audioType = NoAudio;
		exposure = 0.0f;
		xres = 1280;
		yres = 960;
		fps = -1.0f;
		playbackfps = 30.0f;
		fileExt = "mp4";

		stereoType = PanoramaStereoType::Panorama_Mono;
	}

	QString getStereoTypeText() {
		QString typeText = "";
		switch (stereoType)
		{
		case PanoramaStereoType::Panorama_LeftEye:
			typeText = "Left";
			break;
		case PanoramaStereoType::Panorama_RightEye:
			typeText = "Right";
			break;
		case PanoramaStereoType::Panorama_BothEye:
			typeText = "Both";
			break;
		case PanoramaStereoType::Panorama_Mono:
		default:
			typeText = "Mono";
			break;
		}
		return typeText;
	}

	enum PanoramaStereoType
	{
		Panorama_Mono,
		Panorama_LeftEye,
		Panorama_RightEye,
		Panorama_BothEye
	};

	enum InputAudioChannelType
	{
		MixedChannel,
		LeftChannel,
		RightChannel,
		NoAudio
	};

	bool isExistAudio() { return audioName.isEmpty() || audioName.trimmed() == "" ? false : true; }
	
	QString name;
	PanoramaStereoType stereoType;
	QString audioName;
	InputAudioChannelType audioType;
	QString fileDir;			// Input file Dir
	QString filePrefix;			// Input file Prefix
	QString fileExt;			// Input file ext

	float		exposure;			// Capture camera exposure setting
	int			xres;				// Capture X res
	int			yres;				// Capture Y res
	float		fps;				// Capture fps
	float       playbackfps;        // Playback fps
	int			pixel_format;

	bool		valid;

	D360::Capture::CaptureDomain cameraType;

	CameraParameters m_cameraParams; // Camera information from the files
};

typedef unsigned char byte;
typedef struct PanoBuffer {
	byte* bufferPtr;
	uint size;
} PANO_BUFFER, *PANO_BUFFER_PTR;

struct BannerInput
{
	QString bannerFile;
	bool isStereoRight = false;
	bool isVideo = false;
	vec2 quad[4]; // window2pano_1x1
};

struct WeightMapInput
{
	QString weightMapFile;	
};

#endif // STRUCTURES_H
