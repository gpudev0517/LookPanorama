#pragma once

//
// Local
//
#include "CaptureThread.h"
#include "Structures.h"
#include "D360Stitcher.h"
#include "SharedImageBuffer.h"



class CameraModule : public QObject
{
	Q_OBJECT

public:
	explicit CameraModule(int deviceNumber, SharedImageBuffer *sharedImageBuffer, QObject* main = NULL);
	virtual ~CameraModule();

	void qquit();
	void stopCaptureThread();
	void startThreads(bool isReplay = false);

	virtual bool connectToCamera(int width, int height, D360::Capture::CaptureDomain cameraType);
	void snapshot(bool isCalibrate = false);
	IAudioThread * getAudioThread();
	CaptureThread * getCaptureThread();
	bool isConnected() { return isCameraConnected; }

protected:
	QObject* m_Main;
	CaptureThread    *captureThread;
	AudioThread		*audioThread;
	AudioInput		*mic;

	SharedImageBuffer *sharedImageBuffer;

	int m_deviceNumber;
	bool isCameraConnected;
	QString m_Name;

	QThread* captureThreadInstance;

signals:
	void newImageProcessingFlags(struct ImageProcessingFlags imageProcessingFlags);
};