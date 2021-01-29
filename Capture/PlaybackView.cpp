#include "PlayBackView.h"

// Qt
#include <QDesktopWidget>
#include <QKeyEvent>
#include <QMessageBox>



/////////////////////// PlaybackModule
PlaybackModule::PlaybackModule(SharedImageBuffer *sharedImageBuffer, QObject* main) :
sharedImageBuffer(sharedImageBuffer)
, playbackThread(0)
, audioThread(0)
, playbackThreadInstance(0)
, m_Main(main)
{
	m_Name = "PlaybackModule";

	// Initialize internal flag
	isPlaybackConnected = false;

	// Register type
	qRegisterMetaType< struct ThreadStatisticsData >("ThreadStatisticsData");
	qRegisterMetaType<MatBuffer>("MatBufferPtr");
}

PlaybackModule::~PlaybackModule()
{
	qquit();

	if (playbackThread)
	{		
		playbackThread->stop();
		if (m_Main)
		{
			disconnect(playbackThread, SIGNAL(finished(int, QString, int)), m_Main, SLOT(finishedThread(int, QString, int)));
			disconnect(playbackThread, SIGNAL(started(int, QString, int)), m_Main, SLOT(startedThread(int, QString, int)));
			disconnect(playbackThread, SIGNAL(report(int, QString, int)), m_Main, SLOT(reportError(int, QString, int)));
		}
		delete playbackThread;
	}

	if (audioThread)
	{
		audioThread->stop();		
		delete audioThread;
	}
}

bool PlaybackModule::connectToPlayback(int width, int height, D360::Capture::CaptureDomain cameraType)
{
	//
	// Create capture thread
	//
	if (cameraType == D360::Capture::CAPTURE_FILE)
	{
		playbackThread = new PlaybackThread(sharedImageBuffer, D360::Capture::CaptureDomain::CAPTURE_FILE, width, height);
	}
	else if (cameraType == D360::Capture::CAPTURE_DSHOW)
	{
		playbackThread = new PlaybackThread(sharedImageBuffer, D360::Capture::CaptureDomain::CAPTURE_DSHOW, width, height);
	}
	else if (cameraType == D360::Capture::CAPTURE_VIDEO)
	{
		playbackThread = new PlaybackThread(sharedImageBuffer, D360::Capture::CaptureDomain::CAPTURE_VIDEO, width, height);
	}
	else
	{
		playbackThread = new PlaybackThread(sharedImageBuffer, D360::Capture::CaptureDomain::CAPTURE_DSHOW, width, height);
	}

	//
	// Attempt to connect to playback
	// 
	if (playbackThread->connect())
	{

		//
		// Create processing thread
		//

		playbackThreadInstance = new QThread;
		playbackThread->moveToThread(playbackThreadInstance);
		//connect(playbackThread, SIGNAL(error(QString)), this, SLOT(errorString(QString)));
		connect(playbackThreadInstance, SIGNAL(started()), playbackThread, SLOT(process()));
		//connect(playbackThread, SIGNAL(finished()), playbackThreadInstance, SLOT(quit()));
		//connect(playbackThreadInstance, SIGNAL(finished()), playbackThread, SLOT(deleteLater()));
		//connect(playbackThread, SIGNAL(finished(int, QString, int)), playbackThreadInstance, SLOT(deleteLater()));
		if (m_Main) {
			connect(playbackThread, SIGNAL(finished(int, QString, int)), m_Main, SLOT(finishedThread(int, QString, int)));
			connect(playbackThread, SIGNAL(started(int, QString, int)), m_Main, SLOT(startedThread(int, QString, int)));
			connect(playbackThread, SIGNAL(report(int, QString, int)), m_Main, SLOT(reportError(int, QString, int)));
		}

		//
		// Set internal flag and return
		//
		isPlaybackConnected = true;

		return true;
	}
	// Failed to connect to playback
	else
		return false;
}

IAudioThread * PlaybackModule::getAudioThread()
{
	return (IAudioThread*)playbackThread;
}

PlaybackThread* PlaybackModule::getPlaybackThread()
{
	return playbackThread;
}

void PlaybackModule::startThreads(bool isReplay)
{
	if (playbackThreadInstance) {
		if (isReplay)
		{
			if (playbackThreadInstance->isRunning())
			{
				playbackThreadInstance->terminate();
//				playbackThreadInstance->wait();
			}
			playbackThread->reconnect();
		}
		playbackThreadInstance->start();
	}
}

void PlaybackModule::qquit()
{
	//std::cout << "Camera Module Finished " << std::endl;

	if (isPlaybackConnected)
	{
		//
		// Stop playback thread
		//
		//if (playbackThreadInstance && playbackThreadInstance->isRunning()) // commented by B
		if (playbackThreadInstance)
		{
			stopPlaybackThread();
			// After this, on release function, will be delete.
			sharedImageBuffer->removeByDeviceNumber(PLAYBACK_CAMERA_INDEX);
		}
	}
}

void PlaybackModule::stopPlaybackThread()
{
	PANO_N_LOG("About to stop playback thread...");

	playbackThread->stop();
	playbackThread->thread()->terminate();

	sharedImageBuffer->wakeAll(); // This allows the thread to be stopped if it is in a wait-state
	if (!playbackThread->isFinished())
		playbackThread->waitForFinish();

	// Take one frame off a FULL queue to allow the capture thread to finish
	if (playbackThreadInstance)
	{
		delete playbackThreadInstance;
		playbackThreadInstance = NULL;
	}

	PANO_N_LOG("Playback thread successfully stopped.");
}
