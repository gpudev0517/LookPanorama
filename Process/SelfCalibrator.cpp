#include "SelfCalibrator.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "Include/Config.h"

using namespace std;
using namespace cv;

CSelfCalibrator::CSelfCalibrator()
{
	lensType = LensType_Fisheye;
	boardWidth = CHECKBOARD_WIDTH;
	boardHeight = CHECKBOARD_HEIGHT;
}

CSelfCalibrator::~CSelfCalibrator()
{
}

static double computeReprojectionErrors(
	const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors,
	bool isFisheye)
{
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); i++)
	{
		std::vector<cv::Point2f> imagePoints2;
		imagePoints2.resize(objectPoints[i].size());
		if (isFisheye)
		{
			cv::fisheye::projectPoints(Mat(objectPoints[i]), imagePoints2, rvecs[i], tvecs[i],
				cameraMatrix, distCoeffs);
		}
		else
		{
			projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
				cameraMatrix, distCoeffs, imagePoints2);
		}
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners)
{
	corners.resize(0);

	for (int i = 0; i < boardSize.height; i++)
	for (int j = 0; j < boardSize.width; j++)
		corners.push_back(Point3f(float(j*squareSize),
		float(i*squareSize), 0));
}

static bool runCalibration(vector<vector<Point2f> > imagePoints,
	Size imageSize, Size boardSize,
	float squareSize, float aspectRatio,
	int flags, Mat& cameraMatrix, Mat& distCoeffs,
	vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs,
	double& totalAvgErr,
	bool isFisheye)
{
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (flags & CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = aspectRatio;

	distCoeffs = Mat::zeros(8, 1, CV_64F);
	Mat fisheyeDistCoeffs = Mat::zeros(4, 1, CV_64F);

	std::vector<std::vector<cv::Point3f> > objectPoints(1);
	calcChessboardCorners(boardSize, squareSize, objectPoints[0]);

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	double rms;
	if (isFisheye)
	{
		rms = fisheye::calibrate(objectPoints, imagePoints, imageSize, cameraMatrix,
			fisheyeDistCoeffs, rvecs, tvecs, flags | fisheye::CALIB_FIX_SKEW | fisheye::CALIB_FIX_K4);
		distCoeffs = fisheyeDistCoeffs;
	}
	else
	{
		rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
			distCoeffs, rvecs, tvecs, flags | CALIB_FIX_ASPECT_RATIO | CALIB_ZERO_TANGENT_DIST | CALIB_FIX_K4 | CALIB_FIX_K5);
	}

	//printf("RMS error reported by calibrateCamera: %g\n", rms);

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs, isFisheye);

	return ok;
}


static void saveCameraParams(const string& filename,
	Size imageSize, Size boardSize,
	float squareSize, float aspectRatio, int flags,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const vector<float>& reprojErrs,
	const vector<vector<Point2f> >& imagePoints,
	double totalAvgErr)
{
	FileStorage fs(filename, FileStorage::WRITE);

	time_t tt;
	time(&tt);
	struct tm *t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf)-1, "%c", t2);

	fs << "calibration_time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << boardSize.width;
	fs << "board_height" << boardSize.height;
	fs << "square_size" << squareSize;

	if (flags & CALIB_FIX_ASPECT_RATIO)
		fs << "aspectRatio" << aspectRatio;

	if (flags != 0)
	{
		sprintf(buf, "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
		//cvWriteComment( *fs, buf, 0 );
	}

	fs << "flags" << flags;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;
	if (!reprojErrs.empty())
		fs << "per_view_reprojection_errors" << Mat(reprojErrs);

	if (!rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
		for (int i = 0; i < (int)rvecs.size(); i++)
		{
			Mat r = bigmat(Range(i, i + 1), Range(0, 3));
			Mat t = bigmat(Range(i, i + 1), Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
			//*.t() is MatExpr (not Mat) so we can use assignment operator
			r = rvecs[i].t();
			t = tvecs[i].t();
		}
		//cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
		fs << "extrinsic_parameters" << bigmat;
	}

	if (!imagePoints.empty())
	{
		Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (int i = 0; i < (int)imagePoints.size(); i++)
		{
			Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}
}

CameraParameters cameraParams_SelfCalib;

static bool runAndSave(const string& outputFilename,
	const vector<vector<Point2f> >& imagePoints,
	Size imageSize, Size boardSize, float squareSize,
	float aspectRatio, int flags, Mat& cameraMatrix,
	Mat& distCoeffs, bool writeExtrinsics, bool writePoints,
	bool isFisheye, double& outTotalAvgErr)
{
	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(imagePoints, imageSize, boardSize, squareSize,
		aspectRatio, flags, cameraMatrix, distCoeffs,
		rvecs, tvecs, reprojErrs, totalAvgErr,
		isFisheye);
	printf("%s. avg reprojection error = %.2f\n",
		ok ? "Calibration succeeded" : "Calibration failed",
		totalAvgErr);

	cameraMatrix.at<double>(0, 2) -= imageSize.width / 2;
	cameraMatrix.at<double>(1, 2) -= imageSize.height / 2;

	if (!isFisheye)
	{
		distCoeffs.at<double>(2, 0) = distCoeffs.at<double>(4, 0);
		distCoeffs.resize(4, 1);
	}

	cameraParams_SelfCalib.m_lensType = isFisheye ?
		CameraParameters::LensType_opencvLens_Fisheye :
		CameraParameters::LensType_opencvLens_Standard;
	float focalX = cameraMatrix.at<double>(0, 0);
	float focalY = cameraMatrix.at<double>(1, 1);
	float fovX = atan2(imageSize.width / 2, focalX) * 180 / 3.1415927 * 2;
	float fovY = atan2(imageSize.width / 2, focalY) * 180 / 3.1415927 * 2;
	cameraParams_SelfCalib.m_fov = fovX;
	cameraParams_SelfCalib.m_fovy = fovY;
	cameraParams_SelfCalib.m_offset_x = cameraMatrix.at<double>(0, 2);
	cameraParams_SelfCalib.m_offset_y = cameraMatrix.at<double>(1, 2);
	cameraParams_SelfCalib.m_k1 = distCoeffs.at<double>(0, 0);
	cameraParams_SelfCalib.m_k2 = distCoeffs.at<double>(1, 0);
	cameraParams_SelfCalib.m_k3 = distCoeffs.at<double>(2, 0);

	outTotalAvgErr = totalAvgErr;

	if (ok)
		saveCameraParams(outputFilename, imageSize,
		boardSize, squareSize, aspectRatio,
		flags, cameraMatrix, distCoeffs,
		writeExtrinsics ? rvecs : vector<Mat>(),
		writeExtrinsics ? tvecs : vector<Mat>(),
		writeExtrinsics ? reprojErrs : vector<float>(),
		writePoints ? imagePoints : vector<vector<Point2f> >(),
		totalAvgErr);
	return ok;
}

bool CSelfCalibrator::verifyCheckboard(Mat& frame)
{
	Size boardSize;
	boardSize.width = boardWidth;
	boardSize.height = boardHeight;

	detect_pointbuf.clear();

	bool found = findChessboardCorners(frame, boardSize, detect_pointbuf,
		CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

	return found;
}

bool CSelfCalibrator::calibrate(std::vector<Mat> images, unsigned char * viewBuffer, double& outTotalAvgErr)
{
	string outputFilename = "output.yml";

	Size boardSize, imageSize;
	boardSize.width = boardWidth;
	boardSize.height = boardHeight;

	imageSize.width = images[0].cols;
	imageSize.height = images[0].rows;

	float squareSize = 5;
	float aspectRatio = 1.0f;
	int flags = 0;
	Mat cameraMatrix, distCoeffs;

	imagePoints.clear();

	bool writeExtrinsics = true;
	bool writePoints = true;

	for (int i = 0; i < images.size(); i++)
	{
		Mat view(cv::Size(images[i].cols, images[i].rows), CV_8UC3, viewBuffer);
		Mat viewGray;
		images[i].copyTo(view);

		pointbuf.clear();
		cvtColor(view, viewGray, COLOR_BGR2GRAY);

		bool found = findChessboardCorners(view, boardSize, pointbuf,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

		// improve the found corners' coordinate accuracy
		if (found) cornerSubPix(viewGray, pointbuf, Size(11, 11),
			Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

		if (found)
		{
			imagePoints.push_back(pointbuf);
			//drawChessboardCorners(view, boardSize, Mat(pointbuf), found);
		}

		string msg = format("%d/%d", i + 1, images.size());
		printf("Finding corners ... (%d/%d)\n", i + 1, images.size());
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);
		//putText(view, msg, textOrigin, 1, 1, Scalar(0, 255, 0));

		//imshow("Calibration", view);
		//cvWaitKey(0);
	}

	printf("Calibrating camera parameters ...\n");

	bool ret;
	double avgErr = 0;
	if (imagePoints.size() != 0) {
	ret = runAndSave(outputFilename, imagePoints, imageSize,
		boardSize, squareSize, aspectRatio,
		flags, cameraMatrix, distCoeffs,
		writeExtrinsics, writePoints,
		isFisheye(), avgErr);
	outTotalAvgErr = avgErr;
	}
	else
		return false;

	return ret;
}

void CSelfCalibrator::setLensType(bool isFisheye)
{
	if (isFisheye)
		lensType = LensType_Fisheye;
	else
		lensType = LensType_Standard;
}

bool CSelfCalibrator::isFisheye()
{
	return lensType == LensType_Fisheye;
}

void CSelfCalibrator::setBoardSize(int width, int height)
{
	boardWidth = width;
	boardHeight = height;
}

CameraParameters CSelfCalibrator::getCameraParams()
{
	return cameraParams_SelfCalib;
}