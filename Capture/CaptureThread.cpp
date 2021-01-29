

#include <iomanip>
#include <sstream>

#include <QFile>
#include <QDir>
#include "SharedImageBuffer.h"
#include "CaptureThread.h"

// Local
#include "CaptureImageFile.h"
#include "CaptureDShow.h"
#include "CapturePtGrey.h"

#include "CaptureProp.h"
#include "D360Stitcher.h"
#include "D360Parser.h"
#include "QmlMainWindow.h"
#include "define.h"
#include <QDateTime>
#include "include/Config.h"

extern QmlMainWindow* g_mainWindow;

CaptureThread::CaptureThread(SharedImageBuffer *sharedImageBuffer,
	int  deviceNumber,
	D360::Capture::CaptureDomain captureType,
	int  width, int height) : sharedImageBuffer(sharedImageBuffer),
	m_Name("CaptureThread"),
	m_grabbedFrame(ImageBufferData::NONE)
{
	//
	// Save passed parameters
	//
	this->m_deviceNumber = deviceNumber;
	this->width = width;
	this->height = height;
	this->m_captureType = captureType;

	m_isReplay = false;

	// Initialize variables(s)
	init();
}

CaptureThread::~CaptureThread()
{
	disconnect();
	m_grabbedFrame.dispose();
}

void CaptureThread::init()
{
	doStop = false;
	doPause = true;
	doSnapshot = false;
	doCalibrate = false;
	sampleNumber = 0;
	fpsSum = 0;

	fps.clear();

	cap = NULL;

	statsData.averageFPS = 0;
	statsData.nFramesProcessed = 0;
	statsData.nAudioFrames = 0;

	m_isCanGrab = false;
}

void CaptureThread::run()
{
	m_grabbedFrame.clear();
	cap->start();
	cap->reset(m_grabbedFrame);
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();
	CameraInput& camSettings = gasettings->getCameraInput(m_deviceNumber);


	QDir dir(camSettings.fileDir);
	bool dirExists = false;

	if (camSettings.fileDir != "" && dir.exists()) {
		dirExists = true;
	}
	PANO_DEVICE_DLOG("Cam Dir : " + camSettings.fileDir + " " + (dirExists?"true":"false"));

	int camIndex = m_deviceNumber;
	//
	// If its a file device (input) - the camera index should be set to the appropriate value since for
	// file devices device number is offset by D360_FILEDEVICESTART
	// 
	if (camIndex >= D360_FILEDEVICESTART)
	{
		camIndex -= D360_FILEDEVICESTART;
	}

	// wonder if this has effect: we don't need sync now for this mode,
	// and this won't make things work with subset cameras working
	//sharedImageBuffer->sync( m_deviceNumber );

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
	
	bool bIsFirstFrame = true;
	bool isBadFrame = false;

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
		
		//PANO_LOG_ARG("CaptureThread[%1] -- 1", m_deviceNumber);
		if (!bIsFirstFrame)
		{
			//PANO_LOG("CaptureThread -- Pause");
			if ((doSnapshot || doCalibrate) && statsData.nFramesProcessed > 0) {
				//PANO_LOG("CaptureThread -- Snapshot");

				SharedImageBuffer::ImageDataPtr outImage = cap->convertToRGB888(m_grabbedFrame);
				QImage snapshotImg(outImage.mImageY.buffer, outImage.mImageY.width, outImage.mImageY.height, QImage::Format::Format_RGB888);
				QString snapshotPath = sharedImageBuffer->getGlobalAnimSettings()->m_snapshotDir;
				if (snapshotPath[snapshotPath.length() - 1] == '/')
					snapshotPath = snapshotPath.left(snapshotPath.length() - 1);
				QString imgName = QString(snapshotPath + "/cam%1_%2_%3.jpg").ARGN(m_deviceNumber).arg(statsData.nFramesProcessed).arg(CUR_TIME_H);
				if (doCalibrate)
					imgName = QString("cam%1_%2_%3.jpg").ARGN(m_deviceNumber).arg(statsData.nFramesProcessed).arg(CUR_TIME_H);
				snapshotImg.save(imgName, NULL, 100);
				//PANO_LOG("CaptureThread -- Snapshot Saved");
				if (doSnapshot)
				{
					doSnapshot = false;
					//PANO_LOG("CaptureThread -- Snapshoted");
				}
				else {
					doCalibrate = false;
					emit snapshoted(m_deviceNumber);
					//PANO_LOG("CaptureThread -- Calibrated");
				}
			}

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
		
		//PANO_LOG_ARG("CaptureThread[%1] -- 2", m_deviceNumber);
		if (!m_isReplay && bIsFirstFrame)// && m_grabbedFrame.mImageY.buffer)
		{
			emit snapshoted(m_deviceNumber);
		}
		//PANO_LOG_ARG("CaptureThread[%1] -- 3", m_deviceNumber);

		//PANO_LOG_ARG("CaptureThread[%1] -- 4", m_deviceNumber);
		// Capture frame (if available)
		int nGrabCount = 0;
		while (cap->getIncomingType() != D360::Capture::IncomingFrameType::Video)
		{
			if (!cap->grabFrame(m_grabbedFrame))
			{
				if (IS_NODAL_CAMERA_INDEX(m_deviceNumber))
				{
					cap->seekFirst();
					if (skip_firsts == -1)
					{
						skip_firsts = 0;
					}
				}
				else
				{
					//m_isCanGrab = false;
					cap->seekFirst();
					//break;
				}
			}
			else {
				nGrabCount++;
			}

			//if (m_deviceNumber == nFirstViewId)
			{
				if (cap->getIncomingType() == D360::Capture::IncomingFrameType::Audio)
				{
					if (sharedImageBuffer->getStreamer())
					{
						sharedImageBuffer->syncForAudioProcessing(statsData.nAudioFrames);
						void* frame = cap->retrieveAudioFrame();
						if (frame) {
							sharedImageBuffer->getStreamer()->streamAudio(m_deviceNumber, frame);
							statsData.nAudioFrames++;
						}
					}
				}
			}
			//QThread::msleep(1);
		}
		//PANO_LOG_ARG("CaptureThread[%1] -- 5", m_deviceNumber);

		if (!bIsFirstFrame && !isBadFrame && !sharedImageBuffer->syncForVideoProcessing(statsData.nFramesProcessed))
		{
			if (!sharedImageBuffer->isCaptureFinalizing())
				continue;
		}

		if (m_isCanGrab == false) {
			break;
		}

		if (cap->getIncomingType() == D360::Capture::IncomingFrameType::Video)
		{
			//sharedImageBuffer->sync(m_deviceNumber);
			sharedImageBuffer->lockIncomingBuffer(m_deviceNumber);
			if (!cap->retrieveFrame(0, m_grabbedFrame))
			{
				isBadFrame = true;
				continue;
			}
			else
			{
				isBadFrame = false;
				gasettings->m_xres = m_grabbedFrame.mImageY.width;
				gasettings->m_yres = m_grabbedFrame.mImageY.height;
			}
			sharedImageBuffer->unlockIncomingBuffer(m_deviceNumber);
			//gasettings->m_xres = m_grabbedFrame.mImageData->width();
			//gasettings->m_yres = m_grabbedFrame.mImageData->height();
		}
		else {
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

		//PANO_LOG_ARG("CaptureThread[%1] -- 6", m_deviceNumber);

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
		//PANO_LOG_ARG("CaptureThread[%1] -- 7", m_deviceNumber);

		GlobalState& state = sharedImageBuffer->getState(m_deviceNumber);
		//std::cout << "Current Frame " << state.m_curFrame << std::endl;
		cap->setCurFrame(state.m_curFrame);
		if (!singleFrameMode)
			state.m_curFrame++;

		//SharedImageBuffer::ImageDataPtr qgrabbedframe;
		//qgrabbedframe = m_grabbedFrame.mImageData->clone();

		SharedImageBuffer::ImageDataPtr d = m_grabbedFrame;
		d.mFrame = statsData.nFramesProcessed;
		d.msToWait = cap->msToWait();

		statsData.nFramesProcessed++;

		//
		// Add frame to buffer
		//
		//sharedImageBuffer->getByDeviceNumber( deviceNumber )->add( m_grabbedFrame, dropFrameIfBufferFull );
		//sharedImageBuffer->getByDeviceNumber( deviceNumber )->add( qgrabbedframe, dropFrameIfBufferFull );
		//sharedImageBuffer->getByDeviceNumber( deviceNumber )->add( d, dropFrameIfBufferFull );
		//sharedImageBuffer->getByDeviceNumber( m_deviceNumber )->add( d, dropFrameIfBufferFull );
		sharedImageBuffer->setRawImage(m_deviceNumber, d);

		if (gasettings->m_ui == true)
		{
			//
			ImageBufferData newMat = d;

			//if (gasettings->m_stitch == true)
			{
				std::shared_ptr< D360Stitcher> stitcher = sharedImageBuffer->getStitcher();
				if (stitcher)
				{
					stitcher->updateStitchFrame(newMat, camIndex);
				}
			}
		}

		// Pause if first frame is captured
		if (!m_isReplay && bIsFirstFrame && statsData.nFramesProcessed >= 1) {
			emit firstFrameCaptured(START_MODE);
			bIsFirstFrame = false;
		}

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
		gasettings->setRealFps(1000.0 * nGrabCount / captureTime);

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
	
	PANO_DEVICE_LOG("Capture thread running is finished...");
}

void CaptureThread::process()
{
	m_finished = false;
	emit started(THREAD_TYPE_CAPTURE, "", m_deviceNumber);
	run();	
	forceFinish();
}

void CaptureThread::forceFinish()
{
	if (isConnected())
		thread()->terminate();
	if (isConnected() && m_isCanGrab == false) {		
		PANO_DEVICE_LOG("Reading frame has been finished.");
		emit report(THREAD_TYPE_CAPTURE, "EOF", m_deviceNumber);
		emit finished(THREAD_TYPE_CAPTURE, "EOF", m_deviceNumber);
	} else {
		PANO_DLOG("Emit finished signal");
		emit finished(THREAD_TYPE_CAPTURE, "", m_deviceNumber);
	}
	finishWC.wakeAll();
	m_finished = true;
}

bool CaptureThread::connect()
{
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();
	CameraInput& curCameraInput = gasettings->getCameraInput(m_deviceNumber);

    if (m_captureType == D360::Capture::CaptureDomain::CAPTURE_PTGREY)
    {
        QString cameraName = curCameraInput.name;

        CapturePtGrey* capture = new CapturePtGrey(sharedImageBuffer);

        if (!capture->open(m_deviceNumber, cameraName, gasettings->m_xres, gasettings->m_yres, m_captureType))
        {
            std::cout << "Can't Open Camera" << std::endl;
            emit report(THREAD_TYPE_CAPTURE, "Can't Open DirectShow device.", m_deviceNumber);
            delete capture;
            return false;
        }

        width = capture->getProperty(CV_CAP_PROP_FRAME_WIDTH);
        height = capture->getProperty(CV_CAP_PROP_FRAME_HEIGHT);

        curCameraInput.xres = width;
        curCameraInput.yres = height;

        PANO_DEVICE_LOG(QString("Device resolution (Width: %1, Height: %2)").ARGN(width).ARGN(height));

        cap = capture;
    }

	if (m_captureType == D360::Capture::CaptureDomain::CAPTURE_DSHOW || m_captureType == D360::Capture::CaptureDomain::CAPTURE_VIDEO)
	{
		QString cameraName = curCameraInput.name;

		CaptureDShow* capture = new CaptureDShow(sharedImageBuffer);

		if (!capture->open(m_deviceNumber, cameraName, gasettings->m_xres, gasettings->m_yres, gasettings->m_fps, m_captureType))
		{
			std::cout << "Can't Open Camera" << std::endl;
			emit report(THREAD_TYPE_CAPTURE, "Can't Open DirectShow device.", m_deviceNumber);
			delete capture;
			return false;
		}

		capture->setAudio(curCameraInput.audioType);

		width = capture->getProperty(CV_CAP_PROP_FRAME_WIDTH);
		height = capture->getProperty(CV_CAP_PROP_FRAME_HEIGHT);

		curCameraInput.xres = width;
		curCameraInput.yres = height;

		PANO_DEVICE_LOG(QString("Device resolution (Width: %1, Height: %2)").ARGN(width).ARGN(height));

		cap = capture;
	}

	if (m_captureType == D360::Capture::CaptureDomain::CAPTURE_FILE)
	{
		CaptureImageFile *capture = new CaptureImageFile (sharedImageBuffer);

		GlobalState &state = sharedImageBuffer->getState(m_deviceNumber);
		state.m_curFrame = gasettings->m_startFrame;

		//Buffer< cv::Mat >* imageBuffer = sharedImageBuffer->getByDeviceNumber( deviceNumber );
		capture->setImageFileDir(curCameraInput.fileDir);
		capture->setImageFilePrefix(curCameraInput.filePrefix);
		capture->setImageFileExt(curCameraInput.fileExt);
		capture->setCurFrame(state.m_curFrame);

		std::cout << "Capturing From Image " << state.m_curFrame << std::endl;

		if (!capture->open(m_deviceNumber))
		{
			std::cout << "Can't Open File " << std::endl;
			//emit report(THREAD_TYPE_CAPTURE, "Can't Open image file.", m_deviceNumber);
			g_mainWindow->reportError(THREAD_TYPE_CAPTURE, "Can't Open image file.", m_deviceNumber);
			delete capture;
			return false;
		}

		capture->setProperty(CV_CAP_PROP_FPS, gasettings->m_fps);

		//
		// Set resolution
		//

		width = capture->getProperty(CV_CAP_PROP_FRAME_WIDTH);
		height = capture->getProperty(CV_CAP_PROP_FRAME_HEIGHT);
		//float fps = capture->getProperty(CV_CAP_PROP_FPS);

		//gasettings->m_xres = width;
		//gasettings->m_yres = height;
		//gasettings->m_fps = fps;

		cap = capture;
	}

	cap->setSnapshotPath(gasettings->m_snapshotDir);
	m_isCanGrab = true;

	return true;
}

bool CaptureThread::disconnect()
{
	//
	// Camera is connected
	//
	if (cap)
	{
		PANO_DEVICE_LOG("Disconnecting Camera...");
		//
		// Disconnect camera
		//
		delete cap;
		cap = 0;
		return true;
	}
	//
	// Camera is NOT connected
	//
	else
		return false;
}

bool CaptureThread::reconnect()
{
	disconnect();
	init();	

	m_isReplay = true;

	sharedImageBuffer->getStitcher()->initForReplay();

	return connect();
}

void CaptureThread::updateFPS(int timeElapsed)
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

	//
	// Adjust frame playback speed if its loading from file
	// 
/*
	if( m_captureType == CAPTUREFILE )
	{
		GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();
		float fps = 0;
		if (timeElapsed > 0)
			fps = 1000 / timeElapsed;
		if( gasettings->m_playbackfps < fps )
		{
			float sleepms = (1.0/gasettings->m_playbackfps)*1000.0f - timeElapsed ;
			//std::cout << "Ms " << sleepms << std::endl;
			#ifdef _WIN32
			Sleep( sleepms );
			#else
			usleep( sleepms );
			#endif
		}
	}
*/
}

void CaptureThread::waitCapture()
{
	pauseWCMutex.lock();
	pauseWC.wait(&pauseWCMutex, 500);
	pauseWCMutex.unlock();
}

void CaptureThread::wakeCapture()
{
	pauseWCMutex.lock();
	pauseWC.wakeOne();
	pauseWCMutex.unlock();
}


void CaptureThread::stop()
{
	QMutexLocker locker(&doStopMutex);
	doStop = true;
	wakeCapture();
}

void CaptureThread::playAndPause(bool isPause)
{
	QMutexLocker locker(&doPauseMutex);
	doPause = isPause;	
	if (!doPause)
		wakeCapture();
}

void CaptureThread::snapshot(bool isCalibrate)
{
	QMutexLocker locker(&doPauseMutex);
	//if (!doPause)	return;
	if (isCalibrate)
		doCalibrate = true;
	else
		doSnapshot = true;
}

bool CaptureThread::isConnected()
{
	return (cap != NULL);
}

int CaptureThread::getInputSourceWidth()
{
	return cap->getProperty(CV_CAP_PROP_FRAME_WIDTH);
}

int CaptureThread::getInputSourceHeight()
{
	return cap->getProperty(CV_CAP_PROP_FRAME_HEIGHT);
}
#if 0	// Original snapshot function
void CaptureThread::snapshot()
{
	cap->snapshot();
}
#endif
AudioInput * CaptureThread::getMic()
{
	return (AudioInput*)((CaptureDShow*)cap);
}

void CaptureThread::waitForFinish()
{
	finishMutex.lock();
	finishWC.wait(&finishMutex, 1000);
	finishMutex.unlock();
}

bool CaptureThread::isFinished()
{
	return m_finished;
}
