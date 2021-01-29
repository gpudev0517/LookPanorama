#include "LiDARThread.h"

#include <iomanip>
#include <sstream>

#include <QFile>
#include <QDir>

#include "CaptureDevices.h"
#include "CaptureLiDAR.h"

#include "define.h"
#include "TLogger.h"
#include "Include/Config.h"

extern QThread* g_mainThread;


// LiDARThread

LiDARThread::LiDARThread( QObject* main )
	: threadInstance( 0 )
	, m_streamer( 0 )
	, m_Name( "LiDARThread" )
	, m_grabbedFrame( ImageBufferData::NONE )
	, m_Main( main )
{

	//
	// Initialize variables(s)
	//
	doExit = false;
	sampleNumber = 0;
	fpsSum = 0;

	fps.clear();

	cap = NULL;

	statsData.averageFPS = 0;
	statsData.nFramesProcessed = 0;
}

LiDARThread::~LiDARThread()
{
	disconnect();
	stopThread();
	m_grabbedFrame.dispose();
	if( threadInstance )
		FREE_PTR( threadInstance );
}

void LiDARThread::initialize( SharedImageBuffer *sharedImageBuffer )
{
	//
	// Save passed parameters
	//
	this->sharedImageBuffer = sharedImageBuffer;
	m_streamer = sharedImageBuffer->getStreamer();

	this->connect();

	if( cap )
	{
		threadInstance = new QThread();
		this->moveToThread( threadInstance );
		QObject::connect( threadInstance, SIGNAL( started() ), this, SLOT( process() ) );
		QObject::connect( threadInstance, SIGNAL( finished() ), this, SLOT( deleteLater() ) );
		QObject::connect( this, SIGNAL( finished( int, QString, int ) ), threadInstance, SLOT( deleteLater() ) );
		if( m_Main )
		{
			QObject::connect( this, SIGNAL( started( int, QString, int ) ), m_Main, SLOT( startedThread( int, QString, int ) ) );
			QObject::connect( this, SIGNAL( finished( int, QString, int ) ), m_Main, SLOT( finishedThread( int, QString, int ) ) );
		}
	}
}

void LiDARThread::run()
{
	m_grabbedFrame.clear();
	cap->reset( m_grabbedFrame );
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();

	int frameNum = 0;

	// Start timer (used to calculate capture rate)
	t.start();

	while( 1 )
	{
		if( cap == NULL )
			doExit = true;
		if( QThread::currentThread()->isInterruptionRequested() )
		{
			std::cout << "Got signal to terminate" << std::endl;
			doExit = true;
		}
		//
		// Stop thread if doExit = TRUE 
		//
		doExitMutex.lock();
		if( doExit )
		{
			std::cout << "Stop" << std::endl;
			doExit = false;
			doExitMutex.unlock();
			break;
		}
		doExitMutex.unlock();

		//pausehWC.wait(&doPauseMutex);

		// Capture frame (if available)
		if( !cap->grabFrame( m_grabbedFrame ) )
		{
			continue;
		}

		if( !cap->retrieveFrame( 0, m_grabbedFrame ) )
		{
			continue;
		}

		//std::cout << "Current Frame " << state.m_curFrame << std::endl;
		statsData.nFramesProcessed++;

		//
		// Add frame to buffer
		//
		if( sharedImageBuffer->getStreamer() )
		{
			sharedImageBuffer->getStreamer()->streamLiDAR( m_grabbedFrame );
			//emit newAudioFrameReady((void*)frame);
		}

		//std::cout << "Grabbed Frame " << m_grabbedFrame << " - data " << std::endl;
		//
		// Save capture time
		//
		captureTime = t.elapsed();
		t.start();

		//
		// Update statistics
		//
		updateFPS( captureTime );

		//
		// Inform GUI of updated statistics
		//
		if( statsData.nFramesProcessed % 10 == 0 && gasettings->m_ui == true )
		{
			emit updateStatisticsInGUI( statsData );
		}
		Sleep( 1 );
	}

	disconnect();

	//delete m_grabbedFrame.mImageData;

	qDebug() << "Stopping audio thread...";
}

void LiDARThread::process()
{
	m_finished = false;
	emit started( THREAD_TYPE_LIDAR, "", 0 );
	run();
	std::cout << "Lidar Thread - Emit finished signal" << std::endl;
	emit finished( THREAD_TYPE_LIDAR, "", m_deviceNumber );
	finishWC.wakeAll();
	//FREE_PTR(threadInstance);
	m_finished = true;

	this->moveToThread( g_mainThread );
}

bool LiDARThread::connect()
{
	if( sharedImageBuffer->getGlobalAnimSettings()->getLiDARPort() == -1 )
		return false;

	LiDARInput* capture = new LiDARInput( sharedImageBuffer );
	if( !capture->open( sharedImageBuffer->getGlobalAnimSettings()->getLiDARPort() ) )
	{
		std::cout << "Can't Open LiDAR Device." << std::endl;
		delete capture;
		return false;
	}

	PANO_DEVICE_LOG( QString( "Connecting to LiDAR device")  );

	cap = capture;

	return true;
}


bool LiDARThread::disconnect()
{
	if( !cap )	return false;

	std::cout << "Disconnecting LiDAR device " << std::endl;
	delete cap;
	cap = 0;

	return true;
}

void LiDARThread::updateFPS( int timeElapsed )
{
	//
	// Add instantaneous FPS value to queue
	//
	if( timeElapsed > 0 )
	{
		fps.enqueue( (int)1000 / timeElapsed );
		//
		// Increment sample number
		//
		sampleNumber++;
	}
	statsData.instantFPS = ( 1000.0 / timeElapsed );
	//
	// Maximum size of queue is DEFAULT_CAPTURE_FPS_STAT_QUEUE_LENGTH
	//
	if( fps.size() > CAPTURE_FPS_STAT_QUEUE_LENGTH )
		fps.dequeue();

	//
	// Update FPS value every DEFAULT_CAPTURE_FPS_STAT_QUEUE_LENGTH samples
	//
	if( ( fps.size() == CAPTURE_FPS_STAT_QUEUE_LENGTH ) && ( sampleNumber == CAPTURE_FPS_STAT_QUEUE_LENGTH ) )
	{
		//
		// Empty queue and store sum
		//
		while( !fps.empty() )
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
	float fps = 0;
	if( timeElapsed > 0 )
		fps = 1000 / timeElapsed;

	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();

	//
	// Adjust frame playback speed if its loading from file
	// 
	/*
	if( m_captureType == CAPTUREFILE )
	{
	if( gasettings.m_playbackfps < fps )
	{
	float sleepms = (1.0/gasettings.m_playbackfps)*1000.0f - timeElapsed ;
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

void LiDARThread::startThread()
{
	if( threadInstance ) threadInstance->start();
}

void LiDARThread::stopThread()
{
	if( !threadInstance )
		return;
	PANO_DEVICE_LOG( "About to stop LiDAR thread..." );

	doExitMutex.lock();
	doExit = true;
	doExitMutex.unlock();

	if( this->isFinished() == false )
		waitForFinish();

	PANO_DEVICE_LOG( "LiDAR thread successfully stopped." );
}

void LiDARThread::waitForFinish()
{
	finishMutex.lock();
	finishWC.wait( &finishMutex );
	finishMutex.unlock();
}

void LiDARThread::stop()
{
	pause();
}


void LiDARThread::pause()
{
	doPauseMutex.lock();
}

void LiDARThread::start()
{
	doPauseMutex.unlock();
}

bool LiDARThread::isConnected()
{
	return !cap;
}

LiDARInput * LiDARThread::getInputDevice()
{
	return cap;
}

bool LiDARThread::isFinished()
{
	return m_finished;
}