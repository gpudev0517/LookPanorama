#include <iostream>
#include "PlaybackStitcher.h"
#include "define.h"
// #include <QQmlApplicationEngine>
// #include "QmlMainWindow.h"
#include <QThread>
#include <QImage>
#include <QtGlobal>
#include <QOpenGL.h>
#include <stdio.h>  
#include <iostream>
#include <fstream>
#include <string>
#include <QtGui>
#include <QOffscreenSurface>
#include <QMutex>

#include "Buffer.h"
#include "define.h"
#include "Structures.h"
#include "include/Config.h"
#include "ConfigZip.h"

//extern QmlMainWindow* g_mainWindow;

#define ENABLE_LOG 1

using namespace std;

#define CUR_TIME							QTime::currentTime().toString("mm:ss.zzz")
#define ARGN(num)							arg(QString::number(num))

#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

extern QThread* g_mainThread;

//////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////// 

PlaybackStitcher::PlaybackStitcher(SharedImageBuffer *pSharedImageBuffer, QObject* main) : sharedImageBuffer(pSharedImageBuffer)
, m_stitcherThreadInstance(0)
, m_Main(main)
, m_Panorama(NULL)
, m_surface(0)
, m_context(0)
, frameProcN(0)
, m_Name("PlaybackStitcher")
, m_gaSettings(NULL)
{
	initialize();	

	m_finished = true;
}

PlaybackStitcher::~PlaybackStitcher(void)
{
	qquit();

	reset();
}

void PlaybackStitcher::initialize()
{
	frameProcN = 0;
	doStop = false;
	sampleNumber = 0;
	fpsSum = 0;

	fps.clear();
	statsData.averageFPS = 0;
	statsData.nFramesProcessed = 0;
	statsData.elapesedTime = 0;

	doPause = false;

	clear();
}

void PlaybackStitcher::initForReplay()
{
	sharedImageBuffer->initializeForReplay();
	initialize();
}

void PlaybackStitcher::init(QOpenGLContext* context)
{
	initialize();

	reset();

	saveTexture = saveTextureGL;

	m_gaSettings = sharedImageBuffer->getGlobalAnimSettings();

	m_surface = new QOffscreenSurface();
	m_surface->create();

	m_context = new QOpenGLContext();
	QSurfaceFormat format = m_surface->requestedFormat();
	format.setSwapInterval(0);
	format.setSwapBehavior(QSurfaceFormat::SingleBuffer);
	format.setVersion(4, 3);
	format.setProfile(QSurfaceFormat::CompatibilityProfile);
	m_context->setFormat(format);
	m_context->setShareContext(context);
	m_context->create();

	m_context->makeCurrent(m_surface);

	QOpenGLVersionProfile profile;
	profile.setProfile(m_context->surface()->format().profile());
	QOpenGLFunctions* gl = (QOpenGLFunctions*)m_context->functions();
	functions_2_0 = m_context->versionFunctions<QOpenGLFunctions_2_0>();
	functions_4_3 = m_context->versionFunctions<QOpenGLFunctions_4_3_Compatibility>();

	int panoCount = m_gaSettings->isStereo() ? 2 : 1;
	int panoWidth = m_gaSettings->m_panoXRes;
	int panoHeight = m_gaSettings->m_panoYRes * panoCount;

	m_viewProcessors = new SingleViewProcess(sharedImageBuffer, PLAYBACK_CAMERA_INDEX, panoWidth, panoHeight);
	m_viewProcessors->create(gl, functions_2_0, functions_4_3);

	m_Panorama = new GLSLPanorama();		
	m_Panorama->setGL(gl, functions_2_0);
	m_Panorama->initialize(panoWidth, panoHeight, false);

	// Panorama resolution will be 4k * 2k,
	// for stereo mode, the top-down video resolution will be 4k * 4k, (which is w by 2h)
	// for mono mode, the resolution will be 4k * 2k (which is w by h)
	int panoramaFrameBytes = panoWidth * panoHeight * 3 / 2;

	// 
	m_context->doneCurrent();

	isFirstFrame = true;
}

void PlaybackStitcher::qquit()
{
	PANO_LOG("PlaybackStitcher Finished ");
	if (m_stitcherThreadInstance && m_stitcherThreadInstance->isRunning())
		stopStitcherThread();
	releaseStitcherThread();
}

void PlaybackStitcher::releaseStitcherThread()
{
	if (m_stitcherThreadInstance)
	{
		m_stitcherThreadInstance->quit();
		m_stitcherThreadInstance->wait();
		delete m_stitcherThreadInstance;
		m_stitcherThreadInstance = NULL;
	}
}

void PlaybackStitcher::stopStitcherThread()
{
	QMutexLocker locker(&doStopMutex);
	doStop = true;
	sharedImageBuffer->wakeStitcher();
}

void PlaybackStitcher::setup()
{
	releaseStitcherThread();
	
	m_stitcherThreadInstance = new QThread;
	this->moveToThread(m_stitcherThreadInstance);
	connect(m_stitcherThreadInstance, SIGNAL(started()), this, SLOT(process()));
	//connect(processingThread, SIGNAL(finished()), processingThreadInstance, SLOT(quit()));
	//connect(m_stitcherThreadInstance, SIGNAL(finished()), this, SLOT(deleteLater()));
	//connect(this, SIGNAL(finished(int, QString, int)), m_stitcherThreadInstance, SLOT(deleteLater()));
}

void PlaybackStitcher::startThread()
{
	setup();
	m_stitcherThreadInstance->start();
	m_context->doneCurrent();
	m_context->moveToThread(m_stitcherThreadInstance);
}

int PlaybackStitcher::getPanoramaTextureId()
{
	if (m_Panorama == NULL) return -1;
	return m_Panorama->getTargetGPUResourceForInteract();
}

void PlaybackStitcher::updateStitchFrame(ImageBufferData& frame)
{
	QMutexLocker locker(&m_stitchMutex);

	playbackFrameRaw = frame;
	frameProcN++;
	sharedImageBuffer->wakeStitcher();
}

void PlaybackStitcher::doCaptureIncomingFrames()
{
	//PANO_LOG("doCaptureIncomingFrames - 1");
	m_stitchMutex.lock();

	//PANO_LOG_ARG("doCaptureIncomingFrames - 2 [%1]", m_2rgbColorCvt.size());
	m_viewProcessors->uploadTexture(playbackFrameRaw);

// 	QString strSaveName = "D:/colorcvt.png";
// 	saveTexture(m_viewProcessors->getColorCvtBuffer(), 640, 320, strSaveName, m_context->functions(), false);

// 	QImage imgY((uchar*)playbackFrameRaw.mImageY.buffer, playbackFrameRaw.mImageY.width, playbackFrameRaw.mImageY.height, QImage::Format_Grayscale8);
// 	imgY.save("D:/playback(Y).png");
 
// 	QImage imgU((uchar*)playbackFrameRaw.mImageU.buffer, playbackFrameRaw.mImageU.width, playbackFrameRaw.mImageU.height, QImage::Format_Grayscale8);
// 	imgU.save("D:/playback(U).png");
 
// 	QImage imgV((uchar*)playbackFrameRaw.mImageV.buffer, playbackFrameRaw.mImageV.width, playbackFrameRaw.mImageV.height, QImage::Format_Grayscale8);
// 	imgV.save("D:/playback(V).png");

	//PANO_LOG("doCaptureIncomingFrames - 3");
	m_stitchMutex.unlock();

	//PANO_LOG("doCaptureIncomingFrames - 4");
}

void PlaybackStitcher::stitchPanorama()
{
	GPUResourceHandle individualPanoramaTextures[2];
	individualPanoramaTextures[0] = m_viewProcessors->getColorCvtTexture();
	
	m_Panorama->render(individualPanoramaTextures, true);
// 	QString strSaveName = "D:/finalpanorama.png";
// 	saveTexture(m_Panorama->getTargetBuffer(), 640, 320, strSaveName, m_context->functions(), false);
}

void PlaybackStitcher::reset()
{
	if (m_surface)
	{
		if (m_context != QOpenGLContext::currentContext())
			m_context->makeCurrent(m_surface);

		delete[] m_viewProcessors;

		if (m_Panorama)
		{
			delete m_Panorama;
			m_Panorama = NULL;
		}

		m_surface->destroy();
		m_context->doneCurrent();
		delete m_context;
		m_context = NULL;
		delete m_surface;
		m_surface = NULL;
	}
}

void PlaybackStitcher::clear()
{
}

void PlaybackStitcher::process()
{
	m_finished = false;
	emit started(THREAD_TYPE_STITCHER, "", -1);
	run();
	PANO_LOG("About to stop PlaybackStitcher thread...");
	finishWC.wakeAll();	

	if (this == NULL)
	{
		PANO_ERROR("PlaybackStitcher instance unavailable...");
	}

	sharedImageBuffer->wakeAll(); // This allows the thread to be stopped if it is in a wait-state	

	PANO_LOG("Clear PlaybackStitcher instance...");
	clear();	
	PANO_LOG("PlaybackStitcher thread successfully stopped.");		

	emit finished(THREAD_TYPE_STITCHER, "", -1);
	PANO_LOG("PlaybackStitcher - Emit Finished");
	/*while (!this->isFinished())
		Sleep(100);*/
		
	this->moveToThread(g_mainThread);
	if (m_context)
		m_context->moveToThread(g_mainThread);
}

void PlaybackStitcher::run()
{
	// Start timer ( used to calculate processing rate )
	t.start();
	int delay = 1000 / m_gaSettings->m_fps;
	int continueDelay = 5;
	
	while (1)
	{
		if (QThread::currentThread()->isInterruptionRequested())
		{
			doStop = true;
		}

		//
		// Stop thread if doStop = TRUE 
		//
		doStopMutex.lock();
		if (doStop)
		{
			doStop = false;
			doStopMutex.unlock();
			PANO_LOG("received PlaybackStitcher stop command");
			break;
		}
		doStopMutex.unlock();

		//PANO_LOG("Stitcher -- 1");
		sharedImageBuffer->waitStitcher();
		// for extra manipulation, we just restitch
		if (!isFirstFrame && doPause)
		{
			//PANO_LOG("Stitcher -- Pause");
			
			//
			// Pause thread if doPause = TRUE 
			//

			// This "IF" code is needed for fixing the issue that some cameras do NOT show any video screen sometimes, 
			// when firstFrame is captured 
			// (APP should pause when first frame is captured)
			Sleep(100);
			continue;
		}
		else
		{
			//PANO_LOG("Stitcher -- Continue");
			if (frameProcN <= statsData.nFramesProcessed)
			{
				QThread::msleep(continueDelay);
				continue;
			}

			if (m_context != QOpenGLContext::currentContext())
				m_context->makeCurrent(m_surface);
			DoCompositePanorama(m_gaSettings);
			m_context->doneCurrent();
		}
	}

	PANO_LOG("Stopping PlaybackStitcher thread (escaped from run function)...");
}

void PlaybackStitcher::DoCompositePanorama(GlobalAnimSettings* gasettings)
{
	static int avgTime = 0;
	static QTime avgTimer;

	t.restart();
	doCaptureIncomingFrames();
		
	sharedImageBuffer->wakeForVideoPlayback(statsData.nFramesProcessed);
	
	stitchPanorama();

	// Save processing time
	avgTime = avgTimer.elapsed();
	avgTimer.restart();

	// Update statistics
	updateFPS(avgTime);
	statsData.nFramesProcessed++;

	if (isFirstFrame)
		isFirstFrame = false;

	if (gasettings->m_ui == true)
	{
		emit updateStatisticsInGUI(statsData);
	}
}

void PlaybackStitcher::updateFPS(int timeElapsed)
{
	statsData.elapesedTime += timeElapsed; 

	// Add instantaneous FPS value to queue
	if (timeElapsed > 0)
	{
		fps.enqueue((int)1000 / timeElapsed);
		// Increment sample number
		sampleNumber++;
	}

	// Maximum size of queue is DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH
	if (fps.size() > STITCH_FPS_STAT_QUEUE_LENGTH)
		fps.dequeue();

	// Update FPS value every DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH samples
	if ((fps.size() == STITCH_FPS_STAT_QUEUE_LENGTH) && (sampleNumber == STITCH_FPS_STAT_QUEUE_LENGTH))
	{
		// Empty queue and store sum
		while (!fps.empty())
			fpsSum += fps.dequeue();
		// Calculate average FPS
		statsData.averageFPS = 1.0f * fpsSum / STITCH_FPS_STAT_QUEUE_LENGTH;
		// Reset sum
		fpsSum = 0;
		// Reset sample number
		sampleNumber = 0;
	}
}

void PlaybackStitcher::waitForFinish()
{
	finishMutex.lock();
	//finishWC.wait(&finishMutex, 100);
	finishWC.wait(&finishMutex);
	finishMutex.unlock();
}

bool PlaybackStitcher::isFinished()
{
	return m_finished;
}

void PlaybackStitcher::playAndPause(bool isPause)
{
	QMutexLocker locker(&doPauseMutex);
	doPause = isPause;

	if (!isPause)
		t.restart();
}

void PlaybackStitcher::setSeekFrames(int nFrames, bool bIsFirstFrame)
{
	QMutexLocker locker(&doSeekMutex);
	isFirstFrame = bIsFirstFrame;
	statsData.nFramesProcessed = nFrames;
	frameProcN = nFrames;
}

