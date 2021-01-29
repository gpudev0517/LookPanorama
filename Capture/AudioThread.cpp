

#include <iomanip>
#include <sstream>

#include <QFile>
#include <QDir>
//#include <QQmlApplicationEngine>

#include "AudioThread.h"
#include "CaptureDevices.h"

#include "define.h"
#include "TLogger.h"
#include "Include/Config.h"

extern QThread* g_mainThread;

// AudioThread

AudioThread::AudioThread(QObject* main)
	: audioThreadInstance(0)
	, m_streamer(0)
	, m_Name("AudioThread")
	, m_Main(main)
{
    
	//
	// Initialize variables(s)
    //
	doExit		 = false;
    sampleNumber = 0;
    fpsSum		 = 0;

    fps.clear();

	cap = NULL;

    statsData.averageFPS		= 0;
    statsData.nFramesProcessed	= 0;
}

AudioThread::~AudioThread()
{
	disconnect();
	stopAudioThread();
	if (audioThreadInstance)
		FREE_PTR(audioThreadInstance);
}

void AudioThread::initialize(SharedImageBuffer *sharedImageBuffer, int deviceIndex)
{
	//
	// Save passed parameters
	//
	this->sharedImageBuffer = sharedImageBuffer;
	this->m_deviceNumber = deviceIndex;
	m_streamer = sharedImageBuffer->getStreamer();
	m_audioDeviceName = sharedImageBuffer->getGlobalAnimSettings()->getCameraInput(m_deviceNumber).audioName;


	this->connect();

	if (cap)
	{
		audioThreadInstance = new QThread();
		this->moveToThread(audioThreadInstance);
		QObject::connect(audioThreadInstance, SIGNAL(started()), this, SLOT(process()));
		QObject::connect(audioThreadInstance, SIGNAL(finished()), this, SLOT(deleteLater()));
		QObject::connect(this, SIGNAL(finished(int, QString, int)), audioThreadInstance, SLOT(deleteLater()));
		if (m_Main) {
			QObject::connect(this, SIGNAL(started(int, QString, int)), m_Main, SLOT(startedThread(int, QString, int)));
			QObject::connect(this, SIGNAL(finished(int, QString, int)), m_Main, SLOT(finishedThread(int, QString, int)));
		}
	}
}

void AudioThread::run()
{
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();

	int frameNum = 0;
	
	// Start timer (used to calculate capture rate)
	t.start();
	
    while( 1 )
    {
		if (cap == NULL)
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
            doExit  = false;
            doExitMutex.unlock();
            break;
        }
        doExitMutex.unlock();

		//pausehWC.wait(&doPauseMutex);

        // Capture frame (if available)
		int data_present, finished;
		if (cap->read(&data_present, &finished))
		{
			Sleep(1);
			continue;
			// error processing
		}

		if (finished)
		{
			// this won't be happen with mic
			// doExit = true;
		}

		if (!data_present)
			continue;

		AVFrame * frame = cap->getAudioFrame();

		//std::cout << "Current Frame " << state.m_curFrame << std::endl;
		statsData.nFramesProcessed++;

		//
		// Add frame to buffer
		//
		if (frame && sharedImageBuffer->getStreamer()) {
			sharedImageBuffer->getStreamer()->streamAudio(m_deviceNumber, frame);
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
		Sleep(1);
    }
	
	disconnect();
	
	//delete m_grabbedFrame.mImageData;

    qDebug() << "Stopping audio thread...";
}

void AudioThread::process()
{
	m_finished = false;
	emit started(THREAD_TYPE_AUDIO, "", m_deviceNumber);
	run();
	std::cout << "Audio Thread - Emit finished signal" << std::endl;
	emit finished(THREAD_TYPE_AUDIO, "", m_deviceNumber);
	finishWC.wakeAll();
	//FREE_PTR(audioThreadInstance);
	m_finished = true;

	this->moveToThread(g_mainThread);
}

QString getStringFromWString(wstring wstr)
{
	int size = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.length(), 0, 0, 0, NULL);
	char *tmpBuffer = new char[size + 1];
	ZeroMemory(tmpBuffer, size + 1);
	WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), wstr.length(), tmpBuffer, size, 0, NULL);
	QString str(tmpBuffer);
	delete[] tmpBuffer;
	return str;
}

bool AudioThread::connect()
{
#if 0
#endif
	QString inputDevice = "audio=" + m_audioDeviceName;

	AudioMicInput* capture = new AudioMicInput;
	if (capture->open(inputDevice.toLocal8Bit().data()))
	{
		std::cout << "Can't Open Microphone" << std::endl;
		delete capture;
		return false;
	}

	PANO_DEVICE_LOG(QString("Connecting to audio device [%1] (SampleRate=%2, SampleFormat=%3)").arg(m_audioDeviceName).ARGN(capture->getSampleRate()).ARGN(capture->getSampleFmt()));

	cap = capture;

	return true;
}


bool AudioThread::disconnect()
{
	if (!cap)	return false;

	std::cout << "Disconnecting Microphone " << std::endl;
	delete cap;
	cap = 0;

	return true;
}

void AudioThread::updateFPS(int timeElapsed)
{
	//
    // Add instantaneous FPS value to queue
    //
	if( timeElapsed > 0 )
    {
        fps.enqueue( ( int ) 1000/timeElapsed );
		//
        // Increment sample number
        //
		sampleNumber++;
    }
	statsData.instantFPS = ( 1000.0/timeElapsed );
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
        statsData.averageFPS = fpsSum/CAPTURE_FPS_STAT_QUEUE_LENGTH;
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
		fps = 1000/timeElapsed;

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

void AudioThread::startThread()
{
	if (audioThreadInstance) audioThreadInstance->start();
}

void AudioThread::stopAudioThread()
{
	if (!audioThreadInstance)
		return;
	PANO_DEVICE_LOG("About to stop audio thread...");

	doExitMutex.lock();
	doExit = true;
	doExitMutex.unlock();

	if (this->isFinished() == false)
		waitForFinish();

	PANO_DEVICE_LOG("Audio thread successfully stopped.");
}

void AudioThread::waitForFinish()
{
	finishMutex.lock();
	finishWC.wait(&finishMutex);
	finishMutex.unlock();
}

void AudioThread::stop()
{
	pause();
}


void AudioThread::pause()
{
	doPauseMutex.lock();
}

void AudioThread::start()
{
	doPauseMutex.unlock();
}

bool AudioThread::isConnected()
{
    return !cap;
}

AudioInput * AudioThread::getMic()
{
	return (AudioInput*)cap;
}

bool AudioThread::isFinished()
{
	return m_finished;
}