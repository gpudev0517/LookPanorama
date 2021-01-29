#include "CameraView.h"

// Qt
#include <QDesktopWidget>
#include <QKeyEvent>
#include <QMessageBox>



/////////////////////// CameraModule
CameraModule::CameraModule(int deviceNumber, SharedImageBuffer *sharedImageBuffer, QObject* main) :
sharedImageBuffer(sharedImageBuffer)
, captureThread(0)
, audioThread(0)
, captureThreadInstance(0)
, m_Main(main)
{
	// Save Device Number
	this->m_deviceNumber = deviceNumber;
	m_Name = "CameraModule";

	// Initialize internal flag
	isCameraConnected = false;

	// Register type
	qRegisterMetaType< struct ThreadStatisticsData >("ThreadStatisticsData");
	qRegisterMetaType<MatBuffer>("MatBufferPtr");
}

CameraModule::~CameraModule()
{
	qquit();

	if (captureThread)
	{		
		captureThread->stop();
		if (m_Main)
		{
			disconnect(captureThread, SIGNAL(finished(int, QString, int)), m_Main, SLOT(finishedThread(int, QString, int)));
			disconnect(captureThread, SIGNAL(started(int, QString, int)), m_Main, SLOT(startedThread(int, QString, int)));
			disconnect(captureThread, SIGNAL(report(int, QString, int)), m_Main, SLOT(reportError(int, QString, int)));
			disconnect(captureThread, SIGNAL(snapshoted(int)), m_Main, SLOT(finishedSnapshot(int)));
		}
		delete captureThread;
	}

	if (audioThread)
	{
		audioThread->stop();
		delete audioThread;
	}

	sharedImageBuffer->removeCamera(m_deviceNumber);
}


bool CameraModule::connectToCamera(int width, int height, D360::Capture::CaptureDomain cameraType)
{
	//
	// Create capture thread
	//
	if (cameraType == D360::Capture::CAPTURE_FILE)
	//if (m_deviceNumber >= D360_FILEDEVICESTART)
	{
		captureThread = new CaptureThread(sharedImageBuffer, m_deviceNumber, D360::Capture::CaptureDomain::CAPTURE_FILE, width, height);
	}
	else if (cameraType == D360::Capture::CAPTURE_DSHOW)
	{
		if (sharedImageBuffer->getGlobalAnimSettings()->m_hasNodalOfflineVideo && m_deviceNumber == sharedImageBuffer->getGlobalAnimSettings()->m_nodalVideoIndex)
			captureThread = new CaptureThread(sharedImageBuffer, m_deviceNumber, D360::Capture::CaptureDomain::CAPTURE_VIDEO, width, height);
		else
			captureThread = new CaptureThread(sharedImageBuffer, m_deviceNumber, D360::Capture::CaptureDomain::CAPTURE_DSHOW, width, height);
	}
	else if (cameraType == D360::Capture::CAPTURE_VIDEO)
	{
		captureThread = new CaptureThread(sharedImageBuffer, m_deviceNumber, D360::Capture::CaptureDomain::CAPTURE_VIDEO, width, height);
	}
	else
	{
		captureThread = new CaptureThread(sharedImageBuffer, m_deviceNumber, D360::Capture::CaptureDomain::CAPTURE_PTGREY, width, height);
	}

	//
	// Attempt to connect to camera
	// 
	if (captureThread->connect())
	{
		//std::cout << "Starting Camera " << m_deviceNumber << std::endl;

		captureThreadInstance = new QThread;
		captureThread->moveToThread(captureThreadInstance);
		//connect(captureThread, SIGNAL(error(QString)), this, SLOT(errorString(QString)));
		connect(captureThreadInstance, SIGNAL(started()), captureThread, SLOT(process()));
		//connect( captureThread, SIGNAL( finished() ), captureThreadInstance, SLOT( quit()));
		//connect(captureThreadInstance, SIGNAL(finished()), captureThread, SLOT(deleteLater()));
		//connect(captureThread, SIGNAL(finished(int, QString, int)), captureThreadInstance, SLOT(deleteLater()));
		if (m_Main) {
			connect(captureThread, SIGNAL(finished(int, QString, int)), m_Main, SLOT(finishedThread(int, QString, int)));
			connect(captureThread, SIGNAL(started(int, QString, int)), m_Main, SLOT(startedThread(int, QString, int)));
			connect(captureThread, SIGNAL(report(int, QString, int)), m_Main, SLOT(reportError(int, QString, int)));
			connect(captureThread, SIGNAL(snapshoted(int)), m_Main, SLOT(finishedSnapshot(int)));
		}

		//
		// Set internal flag and return
		//
		isCameraConnected = true;
#if 0
		if (sharedImageBuffer->getGlobalAnimSettings().getCameraInput(m_deviceNumber).isExistAudio() &&
			audioDevIndex != -1) {
			audioThread = new AudioThread(sharedImageBuffer);
			//audioThread->initialize(sharedImageBuffer, m_deviceNumber);
			//mic = audioThread->getMic();
		}
#endif
		return true;
	}
	// Failed to connect to camera
	else
		return false;
}

void CameraModule::snapshot(bool isCalibrate)
{
	captureThread->snapshot(isCalibrate);
}

IAudioThread * CameraModule::getAudioThread()
{
	return (IAudioThread*)captureThread;
}

CaptureThread* CameraModule::getCaptureThread()
{
	return captureThread;
}

void CameraModule::startThreads(bool isReplay)
{
	if (captureThreadInstance) {
		if (isReplay)
		{
			if (captureThreadInstance->isRunning())
			{
				captureThreadInstance->terminate();
				captureThreadInstance->wait();
			}
			captureThread->reconnect();
		}
		captureThreadInstance->start();
	}
	//if (audioThread) {
	//	audioThread->initialize(sharedImageBuffer, m_deviceNumber);
	//	audioThread->startThread();
	//}
	//if (processingThreadInstance)
	//	processingThreadInstance->start();

	/*
	D360Stitcher* stitcher = sharedImageBuffer->getStitcher().get();
	GlobalAnimSettings& gasettings = sharedImageBuffer->getGlobalAnimSettings();

	stitcher->setGlobalAnimSettings( gasettings );
	connect( processingThread, SIGNAL( newStitchFrameMat( MatBufferPtr, int, int ) ),
	stitcher, SLOT( updateStitchFrameMat( MatBufferPtr, int, int ) ) );
	*/
}

void CameraModule::qquit()
{
	//std::cout << "Camera Module Finished " << std::endl;

	if (isCameraConnected)
	{
		//
		// Stop capture thread
		//
		//if (captureThreadInstance && captureThreadInstance->isRunning()) // commented by B
		if (captureThreadInstance)
		{
			stopCaptureThread();
			// After this, on release function, will be delete.
			//delete captureThreadInstance;
			//captureThreadInstance = NULL;
		}

		// Automatically start frame processing (for other streams)
		/*
		if( sharedImageBuffer->isSyncEnabledForDeviceNumber( deviceNumber ) )
		sharedImageBuffer->setSyncEnabled( true );
		*/

		// Remove from shared buffer
		sharedImageBuffer->removeByDeviceNumber(m_deviceNumber);
		//
		// Disconnect camera
		//
		/*
		if( captureThread->disconnect() )
		qDebug() << "[" << deviceNumber << "] Camera successfully disconnected.";
		else
		qDebug() << "[" << deviceNumber << "] WARNING: Camera already disconnected.";
		*/
	}
	// Delete UI
	//delete ui;
}

void CameraModule::stopCaptureThread()
{
	PANO_DEVICE_LOG("About to stop capture thread...");

	captureThread->stop();
	captureThread->thread()->terminate();

	sharedImageBuffer->wakeAll(); // This allows the thread to be stopped if it is in a wait-state
	if (!captureThread->isFinished())
		captureThread->waitForFinish();

	// Take one frame off a FULL queue to allow the capture thread to finish
	if (captureThreadInstance)
	{
		delete captureThreadInstance;
		captureThreadInstance = NULL;
	}

	PANO_DEVICE_LOG("Capture thread successfully stopped.");
}
