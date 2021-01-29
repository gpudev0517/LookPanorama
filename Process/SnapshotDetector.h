#pragma once

#include <Windows.h>
#include <QObject>

#include "opencv2\imgproc.hpp"

#include "SelfCalibrator.h"

using namespace cv;

class CSnapshotDetector : public QObject
{
	Q_OBJECT
public:
	CSnapshotDetector();
	virtual ~CSnapshotDetector();

	void clear();
	bool detect(Mat frame);
	void finish();
	void setVerifier(ICheckboardVerifier *verifier);
	void takeSnapshot(Mat frame); // this is called when user pressed the "Take Snapshot" button. add frame to the snapshot list when it has the checkboard
	std::vector<Mat> getSnapshots();
	void registerSnapshot();


private:
	void checkSnapshot(Mat frame, float strength);
	void drawStrength(float strength);
	void foundSnapshot(Mat frame, float strength);
	void clearStatus();

	// optical flow
	Mat flow, frame;
	UMat  flowUmat, prevgray;

	// Output Mat to render
	Mat* pOutputMat;

	// snapshot 
	enum SNAPSHOT_STATUS
	{
		SNAPSHOT_UNKNOWN,
		SNAPSHOT_INBOUND
	};
	enum SNAPSHOT_EVENT
	{
		SNAPSHOT_NONE,
		SNAPSHOT_REGISTERED,
		SNAPSHOT_FOUNDCANDIDATE
	};
	std::vector<Mat> snapshotedList; // list of snapshots
	Mat candidateImg; // latest candidate image
	float candidateScore; // latest candidate score
	SNAPSHOT_STATUS snapshotStatus;
	SNAPSHOT_EVENT snapshotEvent;
	int snapshotDuration;

	ICheckboardVerifier * pVerifier;

	// performance test
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;

signals:
	void fireSnapshotChanged(int snapshotCount);
	void strengthRatioChanged(float strengthRatio);
	void strengthDrawColorChanged(QString color);
};

