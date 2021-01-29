#pragma once
#include <Windows.h>
#include "opencv2\imgproc.hpp"

#include "Structures.h"

using namespace cv;

class ICheckboardVerifier
{
public:
	virtual bool verifyCheckboard(Mat& frame) = 0;
};

class CSelfCalibrator : public QObject, public ICheckboardVerifier
{
	Q_OBJECT
public:
	CSelfCalibrator();
	virtual ~CSelfCalibrator();

	virtual bool verifyCheckboard(Mat& frame);
	bool calibrate(std::vector<Mat> images, unsigned char * viewBuffer, double& outTotalAvgErr);

	void setLensType(bool isFisheye);
	bool isFisheye();

	void setBoardSize(int width, int height);

	CameraParameters getCameraParams();

private:
	enum LensType
	{
		LensType_Standard = 3,
		LensType_Fisheye = 4
	};
	LensType lensType;

	int boardWidth;
	int boardHeight;

	// temp
	std::vector<cv::Point2f> detect_pointbuf;
	std::vector < std::vector<cv::Point2f>> imagePoints;
	std::vector<cv::Point2f> pointbuf;
};

