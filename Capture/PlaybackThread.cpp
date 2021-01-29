

#include <iomanip>
#include <sstream>

#include <QFile>
#include <QDir>
#include "SharedImageBuffer.h"
#include "PlaybackThread.h"

// Local
#include "CaptureImageFile.h"
#include "CaptureDShow.h"

#include "CaptureProp.h"
#include "D360Stitcher.h"
#include "D360Parser.h"
#include "PlaybackStitcher.h"
#include "QmlMainWindow.h"
#include "define.h"
#include <QDateTime>
#include "include/Config.h"

extern QmlMainWindow* g_mainWindow;

PlaybackThread::PlaybackThread(SharedImageBuffer *sharedImageBuffer,
	D360::Capture::CaptureDomain captureType,
	int width, int height) : sharedImageBuffer(sharedImageBuffer),
	m_Name("PlaybackThread"),
	m_grabbedFrame(ImageBufferData::NONE)
{
	//
	// Save passed parameters
	//
	this->width = width;
	this->height = height;
	this->m_captureType = captureType;

	cap = NULL;
	m_isReplay = false;
	m_bIsFirstFrame = false;

	// Initialize variables(s)
	init();
}

PlaybackThread::~PlaybackThread()
{
	disconnect();
	m_grabbedFrame.dispose();
}

void PlaybackThread::init()
{
	doStop = false;
	doPause = true;
	sampleNumber = 0;
	fpsSum = 0;

	fps.clear();

	statsData.averageFPS = 0;
	statsData.nFramesProcessed = 0;
	statsData.nAudioFrames = 0;

	m_isCanGrab = false;
}

void PlaybackThread::run()
{
	m_grabbedFrame.clear();
	cap->start();
	cap->reset(m_grabbedFrame);
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();

	// wonder if this has effect: we don't need sync now for this mode,
	// and this won't make things work with subset cameras working

	// Start timer (used to calculate capture rate)
	t.start();

	// 50fps test
	float fps = gasettings->m_fps; // full speed
	if (gasettings->m_captureType == D360::Capture::CAPTURE_DSHOW)
		fps = fps * 2;
	int intervalms = 1000 / fps;

	QDateTime *curTime = new QDateTime;
	qint64 nFirstMS = 0, nSecondMS = 0, nDiffMS = 0;	

	bool singleFrameMode = false;
	if (gasettings->m_startFrame == -1 && gasettings->m_endFrame == -1)
		singleFrameMode = true;
	
	m_bIsFirstFrame = true;

	int skip_firsts = -1;
	while (1)
	{
		nFirstMS = curTime->currentMSecsSinceEpoch();

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
		
		if (!m_bIsFirstFrame)
		{
			//PANO_LOG("PlaybackThread -- Pause");

			doPauseMutex.lock();
			if (doPause)
			{
				doPauseMutex.unlock();
				QThread::msleep(intervalms);
				waitCapture();
				continue;
			}
			doPauseMutex.unlock();
		}
		//
		// Pause thread if doPause = TRUE 
		//
		
		if (!sharedImageBuffer->syncForVideoPlayback(statsData.nFramesProcessed))
			continue;

		// Capture frame (if available)
		while (cap->getIncomingType() != D360::Capture::IncomingFrameType::Video)
		{
			if (!cap->grabFrame(m_grabbedFrame))
			{
				m_isCanGrab = false;
				break;
			}

			{
				if (cap->getIncomingType() == D360::Capture::IncomingFrameType::Audio)
				{
					sharedImageBuffer->syncForAudioPlayback(statsData.nAudioFrames);
					void* frame = cap->retrieveAudioFrame();
					if (frame) {
						//sharedImageBuffer->getStreamer()->streamAudio(m_deviceNumber, frame);
						statsData.nAudioFrames++;
					}
				}
			}
			//QThread::msleep(1);
		}

		if (m_isCanGrab == false) {
			break;
		}

		if (cap->getIncomingType() == D360::Capture::IncomingFrameType::Video)
		{
			sharedImageBuffer->lockPlaybackBuffer();
			if (!cap->retrieveFrame(0, m_grabbedFrame))
				continue;
			sharedImageBuffer->unlockPlaybackBuffer();
		} else {
			//QThread::msleep(intervalms);
			//QThread::msleep(1);
			continue;
		}

		if (skip_firsts >= 0)
		{
			if (skip_firsts >= 100)
			{
				skip_firsts = -1;
				continue;
			}
			skip_firsts++;
			continue;
		}

		if (cap->isOfflineMode())
		{
			nFirstMS = curTime->currentMSecsSinceEpoch();

			// It should delay if second frame is captured faster than 1000/fps				
			nDiffMS = nFirstMS - nSecondMS; // nFirstMS is the next Milliseconds of nSecondMS
			if (nFirstMS != 0 && nSecondMS != 0 && nDiffMS < intervalms)
			{
				if (nDiffMS >= 0)
				{
					// If the second frame is captured faster than normal (1000/fps)
					Sleep(intervalms - nDiffMS);
				}
			}

			nSecondMS = curTime->currentMSecsSinceEpoch();
		}

		GlobalState& state = sharedImageBuffer->getState(PLAYBACK_CAMERA_INDEX);
		if (cap)
			cap->setCurFrame(state.m_curFrame);
		if (!singleFrameMode)
			state.m_curFrame++;

		//SharedImageBuffer::ImageDataPtr qgrabbedframe;
		//qgrabbedframe = m_grabbedFrame.mImageData->clone();

		SharedImageBuffer::ImageDataPtr d1 = m_grabbedFrame;
		d1.mFrame = statsData.nFramesProcessed;
		if (cap) d1.msToWait = cap->msToWait();

		statsData.nFramesProcessed++;

		//
		// Add frame to buffer
		//
		sharedImageBuffer->setPlaybackRawImage(d1);

		// Pause if first frame is captured
		if (!m_isReplay && m_bIsFirstFrame && statsData.nFramesProcessed >= 1) {
				emit firstFrameCaptured(START_MODE);
				m_bIsFirstFrame = false;
		}

		m_isReplay = false;
		//std::cout << "Grabbed Frame " << m_grabbedFrame << " - data " << std::endl;
		if (gasettings->m_oculus) continue;
		//
		// Save capture time
		//
		captureTime = t.elapsed();
		t.start();

		//
		// Update statistics
		//
		updateFPS(captureTime);

		if (gasettings->m_endFrame != -1 && state.m_curFrame > gasettings->m_endFrame)
		{
			Sleep(100);
			//m_isCanGrab = true;
			// for loop,
			//state.m_curFrame = gasettings->m_startFrame;
		}
		
		//
		// Inform GUI of updated statistics
		//
		if (statsData.nFramesProcessed % 10 == 0 && gasettings->m_ui == true)
		{
			emit updateStatisticsInGUI(statsData);
		}
	}
	delete curTime;
	//disconnect();

	//delete m_grabbedFrame.mImageData;
	
	PANO_LOG("Playback Thread running is finished...");
}

void PlaybackThread::process()
{
	m_finished = false;
	emit started(THREAD_TYPE_PLAYBACK, "", PLAYBACK_CAMERA_INDEX);
	run();	
	forceFinish();
}

void PlaybackThread::forceFinish()
{
	if (isConnected())
		thread()->terminate();
	if (isConnected() && m_isCanGrab == false) {		
		PANO_LOG("Reading frame has been finished for playback.");
		emit report(THREAD_TYPE_PLAYBACK, "EOF", PLAYBACK_CAMERA_INDEX);
		emit finished(THREAD_TYPE_PLAYBACK, "EOF", PLAYBACK_CAMERA_INDEX);
	} else {
		PANO_LOG("Emit finished signal for playback");
		emit finished(THREAD_TYPE_PLAYBACK, "", PLAYBACK_CAMERA_INDEX);
	}
	finishWC.wakeAll();
	m_finished = true;
}

bool PlaybackThread::connect()
{
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();

	if (m_captureType == D360::Capture::CaptureDomain::CAPTURE_DSHOW || m_captureType == D360::Capture::CaptureDomain::CAPTURE_VIDEO)
	{
		CaptureDShow* capture = new CaptureDShow(sharedImageBuffer);

		QString trackFilePath = g_mainWindow->getPlayBackFilePath();

		if (!capture->open(PLAYBACK_CAMERA_INDEX, trackFilePath, gasettings->m_xres, gasettings->m_yres, gasettings->m_fps, m_captureType))
		{
			std::cout << "Can't Open Playback" << std::endl;
			emit report(THREAD_TYPE_PLAYBACK, "Can't Open DirectShow device.", PLAYBACK_CAMERA_INDEX);
			delete capture;
			return false;
		}

		//capture->setAudio(CameraInput::NoAudio);

		int nFrames = capture->getTotalFrames();
		g_mainWindow->setDurationString(nFrames);
		
		width = capture->getProperty(CV_CAP_PROP_FRAME_WIDTH);
		height = capture->getProperty(CV_CAP_PROP_FRAME_HEIGHT);

		PANO_LOG(QString("playback resolution (Width: %1, Height: %2)").ARGN(width).ARGN(height));

		cap = capture;
	}

	m_isCanGrab = true;

	return true;
}

bool PlaybackThread::disconnect()
{
	//
	// Playback is connected
	//
	if (cap)
	{
		PANO_LOG("Disconnecting Playback...");
		//
		// Disconnect Playback
		//
		delete cap;
		cap = 0;
		return true;
	}
	//
	// Playback is NOT connected
	//
	else
		return false;
}

bool PlaybackThread::reconnect()
{
	disconnect();
	init();	

	cap = NULL;
	m_isReplay = true;

	sharedImageBuffer->getPlaybackStitcher()->initForReplay();

	return connect();
}

void PlaybackThread::updateFPS(int timeElapsed)
{
	//
	// Add instantaneous FPS value to queue
	//
	if (timeElapsed > 0)
	{
		fps.enqueue((int)1000 / timeElapsed);
		//
		// Increment sample number
		//
		sampleNumber++;
	}
	statsData.instantFPS = (1000.0 / timeElapsed);
	//
	// Maximum size of queue is DEFAULT_CAPTURE_FPS_STAT_QUEUE_LENGTH
	//
	if (fps.size() > CAPTURE_FPS_STAT_QUEUE_LENGTH)
		fps.dequeue();

	//
	// Update FPS value every DEFAULT_CAPTURE_FPS_STAT_QUEUE_LENGTH samples
	//
	if ((fps.size() == CAPTURE_FPS_STAT_QUEUE_LENGTH) && (sampleNumber == CAPTURE_FPS_STAT_QUEUE_LENGTH))
	{
		//
		// Empty queue and store sum
		//
		while (!fps.empty())
			fpsSum += fps.dequeue();
		//
		// Calculate average FPS
		//
		statsData.averageFPS = fpsSum / CAPTURE_FPS_STAT_QUEUE_LENGTH;
		//statsData.averageFPS = cap->getProperty( CV_CAP_PROP_FPS );
		//std::cout << statsData.averageFPS << std::endl;
		//
		// Reset sum
		//
		fpsSum = 0;
		//
		// Reset sample number
		//
		sampleNumber = 0;
	}
}

void PlaybackThread::waitCapture()
{
	pauseWCMutex.lock();
	pauseWC.wait(&pauseWCMutex, 500);
	pauseWCMutex.unlock();
}

void PlaybackThread::wakeCapture()
{
	pauseWCMutex.lock();
	pauseWC.wakeOne();
	pauseWCMutex.unlock();
}

void PlaybackThread::stop()
{
	QMutexLocker locker(&doStopMutex);
	doStop = true;
	wakeCapture();
}

void PlaybackThread::playAndPause(bool isPause)
{
	QMutexLocker locker(&doPauseMutex);
	doPause = isPause;	
	if (!doPause)
		wakeCapture();
}

void PlaybackThread::setSeekFrames(int nFrames, bool bIsFirstFrame)
{
	QMutexLocker locker(&doSeekMutex);

	m_bIsFirstFrame = bIsFirstFrame;
	cap->seekFrames(nFrames);
	statsData.nFramesProcessed = nFrames;
	GlobalState& state = sharedImageBuffer->getState(PLAYBACK_CAMERA_INDEX);
	state.m_curFrame = nFrames;
	if (!doPause)
		wakeCapture();
}

bool PlaybackThread::isConnected()
{
	return (cap != NULL);
}

int PlaybackThread::getInputSourceWidth()
{
	return cap->getProperty(CV_CAP_PROP_FRAME_WIDTH);
}

int PlaybackThread::getInputSourceHeight()
{
	return cap->getProperty(CV_CAP_PROP_FRAME_HEIGHT);
}

AudioInput * PlaybackThread::getMic()
{
	return (AudioInput*)((CaptureDShow*)cap);
}

void PlaybackThread::waitForFinish()
{
	finishMutex.lock();
	finishWC.wait(&finishMutex, 100);
	finishMutex.unlock();
}

bool PlaybackThread::isFinished()
{
	return m_finished;
}