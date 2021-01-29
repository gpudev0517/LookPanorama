

#include <iomanip>
#include <sstream>

#include <QFile>
#include <QDir>
#include "SharedImageBuffer.h"
#include "BannerThread.h"
#include "CaptureProp.h"
#include "D360Stitcher.h"
#include "D360Parser.h"
//#include "QmlMainWindow.h"
#include "define.h"
#include <QDateTime>

//extern QmlMainWindow* g_mainWindow;

BannerThread::BannerThread(SharedImageBuffer *sharedImageBuffer,
	int bannerId,
	QString fileName,
	int  width, int height, float capFPS) : sharedImageBuffer(sharedImageBuffer),
	m_Name("BannerThread"),
	m_fileName(fileName),
	m_grabbedFrame(ImageBufferData::NONE)
{
	// Save passed parameters
	m_width = width;
	m_height = height;

	this->bannerId = bannerId;
	m_captureFPS = capFPS;

	// Initialize variables(s)
	init();
}


void BannerThread::init()
{
	doStop = false;
	doPause = false;
	fpsSum = 0;

	m_isReplay = false;

	fps.clear();

	cap = NULL;

	statsData.averageFPS = 0;
	statsData.nFramesProcessed = 0;
	statsData.nAudioFrames = 0;

	m_isCanGrab = false;
	m_finished = false;
}

BannerThread::~BannerThread()
{
	disconnect();
	m_grabbedFrame.dispose();
}

void BannerThread::run()
{
	m_grabbedFrame.clear();
	cap->start();
	cap->reset(m_grabbedFrame);

	int intervalms = 1000 / m_captureFPS;
	float fps = m_captureFPS; // full speed
	bool bIsFirstFrame = true;
	t.start();
	int elapsedTime = 0;

	while (true)
	{
		if (QThread::currentThread()->isInterruptionRequested())
		{
			std::cout << "Got signal to terminate" << std::endl;
			doStop = true;
		}
		//
		// Stop thread if doStop = TRUE 
		//
		doStopMutex.lock();
		if (doStop)
		{
			//std::cout << "Stop" << std::endl;
			doStop = false;
			doStopMutex.unlock();
			break;
		}
		doStopMutex.unlock();

		//
		// Pause thread if doPause = TRUE 
		//
		doPauseMutex.lock();
		if (doPause)
		{
			doPauseMutex.unlock();
			QThread::msleep(intervalms);
			continue;
		}
		doPauseMutex.unlock();
		//sharedImageBuffer->syncForVideoProcessing(statsData.nFramesProcessed);

		// Capture frame (if available)
		while (cap->getIncomingType() != D360::Capture::IncomingFrameType::Video)
		{
			if (!cap->grabFrame(m_grabbedFrame))
			{
				m_isCanGrab = false;
				break;
			}

			if (cap->getIncomingType() != D360::Capture::IncomingFrameType::Audio)
				cap->retrieveAudioFrame();
		}

		if (!m_isCanGrab)
			break;

		if (cap->getIncomingType() == D360::Capture::IncomingFrameType::Video)
		{
			cap->retrieveFrame(0, m_grabbedFrame);
		}
		else
		{
			QThread::msleep(intervalms);
			continue;
		}

		statsData.nFramesProcessed++;

		std::shared_ptr<D360Stitcher> stitcher = sharedImageBuffer->getStitcher();
		if (stitcher)
		{
			stitcher->lockBannerMutex();
			stitcher->updateBannerVideoFrame(m_grabbedFrame, bannerId);
			stitcher->unlockBannerMutex();
		}

		// Pause if first frame is captured
		if (!m_isReplay && bIsFirstFrame && statsData.nFramesProcessed >= 2)
		{
			playAndPause(true);
			bIsFirstFrame = false;
			sharedImageBuffer->getStitcher()->restitch();
		}

		elapsedTime = t.elapsed();
		t.restart();

		if (elapsedTime < intervalms)
			QThread::msleep(intervalms - elapsedTime);
	}	

	PANO_LOG("Stopping banner thread...");
	forceFinish();
}

void BannerThread::process()
{	
	m_finished = false;
	sharedImageBuffer->getStitcher()->restitch();
	emit started(THREAD_TYPE_BANNER, "", bannerId);	
}

void BannerThread::forceFinish()
{		
	if (isConnected() && m_isCanGrab == false)
	{
		PANO_DLOG("Reading frame has been finished.");
		emit report(THREAD_TYPE_BANNER, "EOF", bannerId);
		emit finished(THREAD_TYPE_BANNER, "EOF", bannerId);
	}
	else
	{
		PANO_DLOG("Emit finished signal");
		emit finished(THREAD_TYPE_BANNER, "", bannerId);
	}
	finishWC.wakeAll();	
}

bool BannerThread::reconnect()
{
	disconnect();
	init();

	m_isReplay = true;

	sharedImageBuffer->getStitcher()->initForReplay();

	return connect();
}

bool BannerThread::connect()
{
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();

	CaptureDShow* capture = new CaptureDShow(sharedImageBuffer);

	if (!capture->open(bannerId, m_fileName, 0, 0, m_captureFPS, D360::Capture::CaptureDomain::CAPTURE_VIDEO))
	{
		std::cout << "Can't Open Camera" << std::endl;
		emit report(THREAD_TYPE_BANNER, "Can't Open banner video file!", bannerId);
		delete capture;
		return false;
	}

	// Set resolution
	if (m_width != -1)
		capture->setProperty(CV_CAP_PROP_FRAME_WIDTH, m_width);
	else
		m_width = capture->getProperty(CV_CAP_PROP_FRAME_WIDTH);
	if (m_height != -1)
		capture->setProperty(CV_CAP_PROP_FRAME_HEIGHT, m_height);
	else
		m_height = capture->getProperty(CV_CAP_PROP_FRAME_HEIGHT);

	int m_deviceNumber = bannerId;
	PANO_DEVICE_LOG(QString("Video resolution (Width: %1, Height: %2)").ARGN(m_width).ARGN(m_height));

	cap = capture;
	m_isCanGrab = true;

	return true;
}


bool BannerThread::disconnect()
{
	if (cap)
	{	// Camera is connected
		int m_deviceNumber = bannerId;
		PANO_DEVICE_LOG("Disconnecting Camera...");
		// Disconnect camera
		delete cap;
		cap = 0;
		return true;
	}	
	else
	{	// Camera is NOT connected
		return false;
	}
}

void BannerThread::stopBannerThread()
{
	QMutexLocker locker(&doStopMutex);
	doStop = true;
}

void BannerThread::playAndPause(bool isPause)
{
	QMutexLocker locker(&doPauseMutex);
	doPause = isPause;
}

bool BannerThread::isConnected()
{
	return (cap != NULL);
}

int BannerThread::getInputSourceWidth()
{
	return cap->getProperty(CV_CAP_PROP_FRAME_WIDTH);
}

int BannerThread::getInputSourceHeight()
{
	return cap->getProperty(CV_CAP_PROP_FRAME_HEIGHT);
}

void BannerThread::waitForFinish()
{
	finishMutex.lock();
	finishWC.wait(&finishMutex, 100);
	finishMutex.unlock();
}

bool BannerThread::isFinished()
{
	return m_finished;
}