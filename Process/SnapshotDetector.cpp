#include "SnapshotDetector.h"

#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2\highgui.hpp"

#include "include/Config.h"

CSnapshotDetector::CSnapshotDetector() :
pVerifier(0)
{
	clear();
	clearStatus();
}


CSnapshotDetector::~CSnapshotDetector()
{
}

void CSnapshotDetector::clear()
{
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);

	snapshotedList.clear();
	int nSnapshotCount = snapshotedList.size();
	emit fireSnapshotChanged(nSnapshotCount);
}

bool CSnapshotDetector::detect(Mat src)
{
	Mat img;
	int scale = 24;

	int width = src.cols;
	int height = src.rows;
	resize(src, img, Size(width / scale, height / scale));

	// just make current frame gray
	cvtColor(img, img, COLOR_BGR2GRAY);

	float strength = 0.0f;

	// For all optical flow you need a sequence of images.. Or at least 2 of them. Previous                           //and current frame
	//if there is no current frame
	// go to this part and fill previous frame
	//else {
	// img.copyTo(prevgray);
	//   }
	// if previous frame is not empty.. There is a picture of previous frame. Do some                                  //optical flow alg. 

	bool drawFlow = false;
	if (prevgray.empty() == false) {

		// calculate optical flow 
		calcOpticalFlowFarneback(prevgray, img, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
		// copy Umat container to standard Mat
		flowUmat.copyTo(flow);

		// By y += 5, x += 5 you can specify the grid 
		for (int y = 0; y < img.rows; y += 5) {
			for (int x = 0; x < img.cols; x += 5)
			{
				// get the flow from y, x position * 10 for better visibility
				const Point2f flowatxy = flow.at<Point2f>(y, x) * 10;

				strength += sqrt(flowatxy.ddot(flowatxy));
			}
		}

		// fill previous image again
		img.copyTo(prevgray);

	}
	else {
		// fill previous image in case prevgray.empty() == true
		img.copyTo(prevgray);
	}

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

	//
	// We now have the elapsed number of ticks, along with the
	// number of ticks-per-second. We use these values
	// to convert to the number of elapsed microseconds.
	// To guard against loss-of-precision, we convert
	// to microseconds *before* dividing by ticks-per-second.
	//

	ElapsedMicroseconds.QuadPart *= 1000;
	float elapsedTime = ElapsedMicroseconds.QuadPart / Frequency.QuadPart;
	StartingTime = EndingTime;

	float strength_ratio = max(0.0f, strength / SNAPSHOT_MAX_STRENGTH);

	//printf("Elapsed Time: %f\tEntropy: %f\n", elapsedTime, strength_ratio);

	checkSnapshot(src, strength_ratio);

	emit strengthRatioChanged(strength_ratio);


	drawStrength(strength_ratio);

	return false;
}

void CSnapshotDetector::finish()
{
	checkSnapshot(Mat(), 1.0f);
}

void CSnapshotDetector::setVerifier(ICheckboardVerifier *verifier)
{
	pVerifier = verifier;
}

std::vector<Mat> CSnapshotDetector::getSnapshots()
{
	return snapshotedList;
}

void CSnapshotDetector::checkSnapshot(Mat frame, float strength_ratio)
{
	snapshotEvent = SNAPSHOT_NONE;
	switch (snapshotStatus)
	{
	case SNAPSHOT_UNKNOWN:
		if (strength_ratio < SNAPSHOT_THRESHOLD_STRENGTH)
		{
			snapshotStatus = SNAPSHOT_INBOUND;
			foundSnapshot(frame, strength_ratio);
		}
		break;
	case SNAPSHOT_INBOUND:
		if (strength_ratio >= SNAPSHOT_THRESHOLD_STRENGTH)
		{
			snapshotStatus = SNAPSHOT_UNKNOWN;
			if (snapshotDuration > SNAPSHOT_DURATION)
				registerSnapshot();
		}
		else
		{
			snapshotDuration++;
			if (candidateScore > strength_ratio)
			{
				foundSnapshot(frame, strength_ratio);
			}
		}
		break;
	}
}

void CSnapshotDetector::foundSnapshot(Mat frame, float strength_ratio)
{
	snapshotEvent = SNAPSHOT_FOUNDCANDIDATE;
	candidateScore = strength_ratio;
	/*if (frame.cols > 1920)
		resize(frame, candidateImg, cvSize(frame.cols / 2, frame.rows / 2));
	else*/
		candidateImg = frame;
}

void CSnapshotDetector::registerSnapshot()
{
	if (!pVerifier || pVerifier->verifyCheckboard(candidateImg))
	{
		snapshotEvent = SNAPSHOT_REGISTERED;
		Mat snapshotImg = candidateImg.clone();
		snapshotedList.push_back(snapshotImg);
		int nSnapshotCount = snapshotedList.size();
		emit fireSnapshotChanged(nSnapshotCount);
	}
	clearStatus();
}

void CSnapshotDetector::drawStrength(float strength_ratio)
{
	QString color = "";
	if (strength_ratio < SNAPSHOT_THRESHOLD_STRENGTH)
	{
		if (snapshotEvent == SNAPSHOT_FOUNDCANDIDATE)
			color = "foundColor";
		else
			color = "stableColor";
	}
	else
		color = "normalColor";

	emit strengthDrawColorChanged(color);
}

void CSnapshotDetector::clearStatus()
{
	candidateScore = SNAPSHOT_MAX_STRENGTH;
	snapshotStatus = SNAPSHOT_UNKNOWN;
	snapshotDuration = 0;
}

void CSnapshotDetector::takeSnapshot(Mat frame)
{
	if (pVerifier->verifyCheckboard(frame))
	{
		//onSnapshotTaken(frame);
	}
}