extern "C++" {
#include "scy/idler.h"
#include "signaler.h"
}

#include "StreamProcess.h"
#include "define.h"
#include "include/Config.h"
#include "TLogger.h"

//#include <QQmlApplicationEngine>
#include <QDir>

extern QThread* g_mainThread;
static scy::Signaler*	gPtrwebRTC_Signaler = nullptr;

StreamProcess::StreamProcess(SharedImageBuffer *sharedImageBuffer, bool toFile, QObject* main) 
:streamingThreadInstance(0),
m_isFile(toFile)
,m_Name("StreamProcess")
,m_Main(NULL)
{
	if (main)	setMain(main);
	m_iswebRTC	= false;

	//
	// Initialize variables(s)
	//
	
	init();

	qRegisterMetaType<MatBufferPtr>("MatBufferPtr");
	qRegisterMetaType<RawImagePtr>("RawImagePtr");

	m_sharedImageBuffer = sharedImageBuffer;	
}

void StreamProcess::init()
{
	doExit = false;
	doPause = false;
	sampleNumber = 0;
	fpsSum = 0;

	m_isOpened = false;
	m_audioInputCount = 0;
	m_audioProcessedCount = 0;
	m_videoInputCount = 0;
	m_videoProcessedCount = 0;
	m_lidarInputCount = 0;
	m_lidarProcessedCount = 0;
	m_audioFrame = 0;
	m_audioChannelCount = 0;

	fps.clear();

	statsData.averageFPS = 0;
	statsData.nFramesProcessed = 0;
}

StreamProcess::~StreamProcess()
{
	stopStreamThread();
}

std::string sampleDataDir(const std::string& file);

void StreamProcess::run()
{
	int frameNum = 0;

	t.start();
	int delay = 1000 / m_sharedImageBuffer->getGlobalAnimSettings()->m_fps;
	int continueCount = 0;
	int continueDelay = 1;
	captureTime = 0;

	while (1)
	{
		if (QThread::currentThread()->isInterruptionRequested())
		{
			std::cout << "Got signal to terminate" << std::endl;
			doExit = true;
		}
		//
		// Stop thread if doExit = TRUE 
		//
		doExitMutex.lock();
		if (doExit)
		{			
			doExit = false;
			doExitMutex.unlock();
			break;
		}
		doExitMutex.unlock();

		doPauseMutex.lock();
		if (doPause)
		{
			doPauseMutex.unlock();			
			continue;
		}
		doPauseMutex.unlock();

		Sleep(1);

		// Start timer (used to calculate capture rate)
		//t.start();

		/*QMap<int, void*> audioFrames = getAudioFrameSeq();
		
		if (audioFrames.size() != 0)
		{
			//audioFrameMutex.lock();
			if (m_audioInputCount > m_audioProcessedCount)
			{
				if (m_audioInputCount > m_audioProcessedCount + 1)
				{
					std::cout << "a"; // audio frame skipped
				}
				m_audioProcessedCount = m_audioInputCount;
				//audioFrameMutex.lock();
				if (m_sharedImageBuffer->getGlobalAnimSettings()->m_captureType == D360::Capture::CaptureDomain::CAPTURE_DSHOW)
					m_Stream.StoreAudioFrame(audioFrames.first());
				else
					m_Stream.StoreAudioFrame(audioFrames);
				//audioFrameMutex.unlock();

				//m_audioReadyFrames.clear();
				m_sharedImageBuffer->wakeForAudioProcessing(m_audioProcessedCount - 1);
			}
			//audioFrameMutex.unlock();
		}*/
		audioFrameMutex.lock();
		QMap<int, void*> audioFrames = m_audioReadyFrames;
		audioFrameMutex.unlock();
		if (m_audioInputCount > m_audioProcessedCount)
		{
			if (m_audioInputCount > m_audioProcessedCount + 1)
			{
				std::cout << "a"; // audio frame skipped
			}
			m_audioProcessedCount = m_audioInputCount;
			//audioFrameMutex.lock();
			if (m_sharedImageBuffer->getGlobalAnimSettings()->m_captureType == D360::Capture::CaptureDomain::CAPTURE_DSHOW)
				m_Stream.StoreAudioFrame(audioFrames.first());
			else
				m_Stream.StoreAudioFrame(audioFrames);
			//audioFrameMutex.unlock();
			m_sharedImageBuffer->wakeForAudioProcessing(m_audioProcessedCount - 1);
		}
		
		// Streaming for exporting the AUDIO to the file
		m_Stream.StreamAudio();


		lidarFrameMutex.lock();
		ImageBufferData lidarFrame = m_LiDARFrame;
		lidarFrameMutex.unlock();
		if( m_lidarInputCount > m_lidarProcessedCount )
		{
			m_lidarProcessedCount = m_lidarInputCount;
			m_LiDARStream.StreamLiDARFrame( lidarFrame.mImageY.buffer, lidarFrame.mImageY.width, lidarFrame.mImageY.height );
			std::cout << "l"; // lidar frame processed
		}


		videoFrameMutex.lock();
		//if (m_videoInputCount <= m_videoProcessedCount || m_Panoramas.size() == 0)
		if (m_videoInputCount <= m_videoProcessedCount || m_Panorama == NULL)
		{
			videoFrameMutex.unlock();
			continueCount++;
			QThread::msleep(continueDelay);
			continue;
		}
		else if (m_videoInputCount > m_videoProcessedCount + 1)
		{
			std::cout << " "; // video frame skipped
		}
		m_videoProcessedCount = m_videoInputCount;
		//m_videoProcessedCount++;

		//if (captureTime + continueDelay * continueCount < delay)
		//	QThread::msleep(delay - captureTime - continueDelay * continueCount);

		continueCount = 0;

		// Should do m_stream.close & m_stream.initialize for offline videos if splitMins was set
		if (m_sharedImageBuffer->getGlobalAnimSettings()->m_splitMins > 0.0f) {

			int totalFramesPerSplitMin = m_sharedImageBuffer->getGlobalAnimSettings()->getFps() * m_sharedImageBuffer->getGlobalAnimSettings()->m_splitMins * 60;
			if (statsData.nFramesProcessed > 0
				&& statsData.nFramesProcessed % totalFramesPerSplitMin == 0)
			{
				disconnect();
				int nSerialNumber = statsData.nFramesProcessed / totalFramesPerSplitMin;
				if (nSerialNumber > 0)
					initializeStream(nSerialNumber);
			}
		}
		// Capture frame (if available)
		// Streaming for exporting the VIDEO to the file
		m_Stream.StreamFrame(m_Panorama, statsData.nFramesProcessed, m_width, m_height, !m_isFile);
		/*QImage frontImage = m_Panoramas.front();
		m_Panoramas.pop_front();
		m_Stream.StreamFrame(frontImage.bits(), statsData.nFramesProcessed, m_width, m_height, !m_isFile);*/
		videoFrameMutex.unlock();
		std::cout << "-"; // video frame processed

		statsData.nFramesProcessed++;
		
		//std::cout << "Grabbed Frame " << m_grabbedFrame << " - data " << std::endl;
		//
		// Save capture time
		//
		captureTime = t.elapsed();
		//PANO_LOG(QString("Streaming time: %1, %2").ARGN(captureTime).arg(CUR_TIME));
		t.restart();

		//
		// Update statistics
		//
		updateFPS(captureTime);


		//
		// Inform GUI of updated statistics
		//
		if (statsData.nFramesProcessed % 10 == 0)
		{
			emit updateStatisticsInGUI(statsData);
		}
		if (statsData.nFramesProcessed % 30 == 0)
			;// printf("Streaming %d\n", statsData.nFramesProcessed);
	}

	disconnect();

	//delete m_grabbedFrame.mImageData;

	PANO_LOG ("Stopping streaming thread...");
}

void StreamProcess::process()
{
	try {
		m_finished = false;	
		emit started(THREAD_TYPE_STREAM, QString::number(m_nCurrentSplitSerialNumber), -1);

//#if USE_SCY_LIB
		if (m_iswebRTC)
		{
			//std::string sourceFile("test.mp4");
			QString sourceFile = QDir::currentPath() + QLatin1String("/test.mp4");

			//scy::Signaler app(m_outFileName.toUtf8().constData(), SERVER_PORT);
			//app.startStreaming(sourceFile, false);
			if (!gPtrwebRTC_Signaler)
			{
				gPtrwebRTC_Signaler = new scy::Signaler(m_outFileName.toUtf8().constData(), SERVER_PORT);
				//gPtrwebRTC_Signaler->startStreaming(sourceFile, false);
				gPtrwebRTC_Signaler->startStreaming(sourceFile.toUtf8().constData(), false);
			}

			//// Process WebRTC threads on the main loop.
			auto rtcthread = rtc::Thread::Current();
			scy::Idler rtc([rtcthread]() {
				rtcthread->ProcessMessages(3);
			});

			int frameNum = 0;

			t.start();
			int delay = 1000 / m_sharedImageBuffer->getGlobalAnimSettings()->m_fps;
			int continueCount = 0;
			int continueDelay = 1;
			captureTime = 0;

			while (true)
			{
				Sleep(5);
				if (QThread::currentThread()->isInterruptionRequested())
				{
					std::cout << "Got signal to terminate" << std::endl;
					doExit = true;
				}
				//
				// Stop thread if doExit = TRUE 
				//
				doExitMutex.lock();
				if (doExit)
				{
					//doExit = false;
					doExitMutex.unlock();
					//break;
					this->moveToThread(g_mainThread);
					continue;
				}
				doExitMutex.unlock();

				doPauseMutex.lock();
				if (doPause)
				{
					doPauseMutex.unlock();
					continue;
				}
				doPauseMutex.unlock();

				try
				{
					//uv_run(app.loop, UV_RUN_ONCE);
					if (gPtrwebRTC_Signaler)
					{
						uv_run(gPtrwebRTC_Signaler->loop, UV_RUN_ONCE);
					}
				}
				catch (...) {
					PANO_LOG("Streaming Server Error!");
					break;
				}

				audioFrameMutex.lock();
				QMap<int, void*> audioFrames = m_audioReadyFrames;
				audioFrameMutex.unlock();
				if (m_audioInputCount > m_audioProcessedCount)
				{
					if (m_audioInputCount > m_audioProcessedCount + 1)
					{
						std::cout << "a"; // audio frame skipped
					}
					m_audioProcessedCount = m_audioInputCount;
					//audioFrameMutex.lock();
					if (m_sharedImageBuffer->getGlobalAnimSettings()->m_captureType == D360::Capture::CaptureDomain::CAPTURE_DSHOW)
						m_Stream.StoreAudioFrame(audioFrames.first());
					else
						m_Stream.StoreAudioFrame(audioFrames);
					//audioFrameMutex.unlock();
					m_sharedImageBuffer->wakeForAudioProcessing(m_audioProcessedCount - 1);
				}

				// Streaming for exporting the AUDIO to the file
				// comment 2017.08.03  ??? 
				//m_Stream.StreamAudio();

				videoFrameMutex.lock();
				//if (m_videoInputCount <= m_videoProcessedCount || m_Panoramas.size() == 0)
				if (m_videoInputCount <= m_videoProcessedCount || m_Panorama == NULL)
				{
					videoFrameMutex.unlock();
					continueCount++;
					QThread::msleep(continueDelay);
					continue;
				}
				else if (m_videoInputCount > m_videoProcessedCount + 1)
				{
					std::cout << " "; // video frame skipped
				}
				m_videoProcessedCount = m_videoInputCount;
				//m_videoProcessedCount++;

				//if (captureTime + continueDelay * continueCount < delay)
				//	QThread::msleep(delay - captureTime - continueDelay * continueCount);

				continueCount = 0;

				// Should do m_stream.close & m_stream.initialize for offline videos if splitMins was set
				if (m_sharedImageBuffer->getGlobalAnimSettings()->m_splitMins > 0.0f) {

					int totalFramesPerSplitMin = m_sharedImageBuffer->getGlobalAnimSettings()->getFps() * m_sharedImageBuffer->getGlobalAnimSettings()->m_splitMins * 60;
					if (statsData.nFramesProcessed > 0
						&& statsData.nFramesProcessed % totalFramesPerSplitMin == 0)
					{
						disconnect();
						int nSerialNumber = statsData.nFramesProcessed / totalFramesPerSplitMin;
						if (nSerialNumber > 0)
							initializeStream(nSerialNumber);
					}
				}
				// Capture frame (if available)
				// Streaming for exporting the VIDEO to the file
				//m_Stream.StreamFrame(m_Panorama, statsData.nFramesProcessed, m_width, m_height, !m_isFile);

				//AVFrame* asyncOneFrame = m_Stream.webRTC_makeStreamFrame(m_Panorama, statsData.nFramesProcessed, m_width, m_height, !m_isFile);
				//app.webRTC_OneFrameStreaming(asyncOneFrame);
				if (gPtrwebRTC_Signaler)
				{
					AVFrame* asyncOneFrame = m_Stream.webRTC_makeStreamFrame(m_Panorama, statsData.nFramesProcessed, m_width, m_height, !m_isFile);
					gPtrwebRTC_Signaler->webRTC_OneFrameStreaming(asyncOneFrame);
				}

				/*QImage frontImage = m_Panoramas.front();
				m_Panoramas.pop_front();
				m_Stream.StreamFrame(frontImage.bits(), statsData.nFramesProcessed, m_width, m_height, !m_isFile);*/
				videoFrameMutex.unlock();
				std::cout << "-"; // video frame processed

				statsData.nFramesProcessed++;

				//std::cout << "Grabbed Frame " << m_grabbedFrame << " - data " << std::endl;
				//
				// Save capture time
				//
				captureTime = t.elapsed();
				//PANO_LOG(QString("Streaming time: %1, %2").ARGN(captureTime).arg(CUR_TIME));
				t.restart();

				//
				// Update statistics
				//
				updateFPS(captureTime);

				//
				// Inform GUI of updated statistics
				//
				if (statsData.nFramesProcessed % 10 == 0)
				{
					emit updateStatisticsInGUI(statsData);
				}
				if (statsData.nFramesProcessed % 30 == 0)
					;// printf("Streaming %d\n", statsData.nFramesProcessed);
			}
			disconnect();

			PANO_LOG("Stopping streaming thread...");
		}
		else
//#endif
		{
			run();
		}

		PANO_LOG("Streaming Thread - Emit finished signal");
	} catch(...) { 
		PANO_LOG("Streaming Thread - Error finished!");
	}
		
	//if (streamingThreadInstance != NULL)
	//	streamingThreadInstance->exit();

	this->moveToThread(g_mainThread);

	emit finished(THREAD_TYPE_STREAM, "", -1);

	finishWC.wakeAll();
}


bool StreamProcess::disconnect()
{
	if (!m_isOpened)
	{
		return false;
	}
	PANO_LOG("Close stream...");
	
	m_Stream.Close();
	m_audioReadyFrames.clear();
	m_audioFrames.clear();
	PANO_LOG("Stream closed");
		
	m_isOpened = false;

	return true;
}

void StreamProcess::updateFPS(int timeElapsed)
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
	if (fps.size() > STREAMING_FPS_STAT_QUEUE_LENGTH)
		fps.dequeue();

	//
	// Update FPS value every DEFAULT_CAPTURE_FPS_STAT_QUEUE_LENGTH samples
	//
	if ((fps.size() == STREAMING_FPS_STAT_QUEUE_LENGTH) && (sampleNumber == STREAMING_FPS_STAT_QUEUE_LENGTH))
	{
		//
		// Empty queue and store sum
		//
		while (!fps.empty())
			fpsSum += fps.dequeue();
		//
		// Calculate average FPS
		//
		statsData.averageFPS = fpsSum / STREAMING_FPS_STAT_QUEUE_LENGTH;
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



void StreamProcess::stopStreamThread()
{
	if (!streamingThreadInstance)
		return;
	PANO_LOG("About to stop Stream thread...");
	doExitMutex.lock();
	doExit = true;
	doExitMutex.unlock();

	if (!m_iswebRTC)
	{
		if (!this->isFinished())
			this->waitForFinish();

		if (streamingThreadInstance)
		{
			delete streamingThreadInstance;
			streamingThreadInstance = NULL;
		}
	}
	
	PANO_LOG("Stream thread successfully stopped.");
}

void StreamProcess::waitForFinish()
{
	finishMutex.lock();
	finishWC.wait(&finishMutex);
	finishMutex.unlock();
}


bool StreamProcess::initialize(bool toFile, QString outFileName, int width, int height, int fps, int channelCount, 
	AVSampleFormat sampleFmt, int srcSampleRate, int sampleRate, int audioLag, int videoCodec, int audioCodec, int crf)
{
	m_Panorama = NULL;
	m_isOpened = false;
	m_audioInputCount = 0;
	m_audioProcessedCount = 0;
	m_videoInputCount = 0;
	m_videoProcessedCount = 0;
	m_lidarInputCount = 0;
	m_lidarProcessedCount = 0;
	m_audioFrame = 0;

	m_toFile = toFile;
	m_outFileName = outFileName;
	m_width = width;
	m_height = height;
	m_fps = fps;
	m_channelCount = channelCount;
	m_sampleFmt = sampleFmt;
	m_srcSampleRate = srcSampleRate;
	m_sampleRate = sampleRate;
	m_audioLag = audioLag;
	m_videoCodec = videoCodec;
	m_audioCodec = audioCodec;
	m_crf = crf;	

	return initializeStream(0);
}

bool StreamProcess::initializeStream(int nSerialNumber)
{
	if (m_outFileName.isEmpty())
	{
		return false;
	}
	if (nSerialNumber == 0)
		init(); // should reset again if (splitMins > 0.f)

	int channels = m_channelCount;
	if (m_sharedImageBuffer->getGlobalAnimSettings()->m_captureType == D360::Capture::CaptureDomain::CAPTURE_DSHOW)
	{
		//if (channels > 2)
		//	channels = 2;
	}
		

	m_nCurrentSplitSerialNumber = nSerialNumber;
	
	QString strFileName = m_outFileName;
	// should add serial number to offline video filename if splitMins is more than 1
	if (m_toFile && m_outFileName != ""
		&& m_sharedImageBuffer->getGlobalAnimSettings()->m_splitMins > 0.0f) {
		int nFilenameIndex = m_outFileName.lastIndexOf(".");
		QString strFileNamePrefix = m_outFileName.left(nFilenameIndex);
		QString strFileExtension = m_outFileName.mid(nFilenameIndex, m_outFileName.length() - nFilenameIndex);
		strFileName = strFileNamePrefix.append("_" + QString::number(nSerialNumber)) + strFileExtension;
	}
	//strFileName = QString( "F:/strFileName.avi" );
	int ret = m_Stream.Initialize( strFileName.toUtf8().constData(), m_width, m_height, m_fps, channels, m_sampleFmt, m_srcSampleRate, m_sampleRate, m_audioLag, m_toFile, m_videoCodec, m_audioCodec, false, m_crf );
	if (strFileName != "")
	{
		int portIndex = strFileName.lastIndexOf( ":" );
		int portNum = strFileName.right( strFileName.length() - portIndex -1 ).toInt();
		QString lidarPortNum;
		lidarPortNum.setNum( portNum + 1 );
		QString lidarURL = strFileName.left( portIndex + 1 ) + lidarPortNum;
		//lidarURL = QString( "F:/lidarStream.avi" );
		m_LiDARStream.Initialize( lidarURL.toUtf8().constData(), LIDAR_STREAM_WIDTH, LIDAR_STREAM_HEIGHT, m_fps, 0, m_sampleFmt, m_srcSampleRate, m_sampleRate, m_audioLag, m_toFile, m_videoCodec, AV_CODEC_ID_NONE, true, m_crf );
	}
	if (ret >= 0)
	{
		if (m_sharedImageBuffer->getGlobalAnimSettings()->m_captureType == D360::Capture::CaptureDomain::CAPTURE_DSHOW)
			m_Stream.setCaptureType(D360::Capture::CaptureDomain::CAPTURE_DSHOW);

		// Get expected audio channel count for output video
		m_audioChannelCount = m_sharedImageBuffer->getGlobalAnimSettings()->getAudioChannelCount();

		if (streamingThreadInstance == NULL)
		{
			streamingThreadInstance = new QThread();
			//if (!m_iswebRTC)
			{
				this->moveToThread(streamingThreadInstance);
			}
			connect(streamingThreadInstance, SIGNAL(started()), this, SLOT(process()));
			//connect(streamingThreadInstance, SIGNAL(finished()), this, SLOT(deleteLater()));
			//connect(this, SIGNAL(finished(int, QString, int)), streamingThreadInstance, SLOT(deleteLater()));
			if (m_Main) {
				connect(this, SIGNAL(finished(int, QString, int)), m_Main, SLOT(finishedThread(int, QString, int))/*, Qt::DirectConnection*/);
				connect(this, SIGNAL(started(int, QString, int)), m_Main, SLOT(startedThread(int, QString, int)));
			}
			streamingThreadInstance->start();
		}

		m_isOpened = true;
		return true;
	}

	return false;
}

void StreamProcess::streamPanorama(unsigned char* panorama)
{
	if (!m_isOpened) return;
	videoFrameMutex.lock();
	m_Panorama = panorama;
	//m_Panoramas.push_back(QImage(panorama, m_width, m_height, QImage::Format::Format_RGB888));
	m_videoInputCount++;
	videoFrameMutex.unlock();
}

void StreamProcess::streamAudio(int devNum, void* audioFrame)
{
	if (!m_isOpened) return;
	audioFrameMutex.lock();
	//m_audioFrame = audioFrame;
	m_audioFrames[devNum] = audioFrame;

	CameraInput::InputAudioChannelType audioType = m_sharedImageBuffer->getGlobalAnimSettings()->getCameraInput(devNum).audioType;

	if (audioType == CameraInput::NoAudio) {
		audioFrameMutex.unlock();
		return; // Error: received audio frame from disabled device
	}

	if (m_audioFrames.size() == m_audioChannelCount)
	{
		m_audioReadyFrames = m_audioFrames;
		m_audioFrames.clear();
		m_audioInputCount++;
	}
	audioFrameMutex.unlock();
}

/*void StreamProcess::streamAudio(int devNum, void* audioFrame)
{
	CameraInput::InputAudioChannelType audioType = m_sharedImageBuffer->getGlobalAnimSettings()->cameraSettingsList()[devNum].audioType;
	if (audioType == CameraInput::NoAudio)
	{
		return; // Error: received audio frame from disabled device
	}

	audioFrameMutex.lock();
	m_audioFrames[devNum].push_back(audioFrame);
	audioFrameMutex.unlock();
}*/

void StreamProcess::streamLiDAR( ImageBufferData& frame )
{
	if( !m_isOpened ) return;
	lidarFrameMutex.lock();
	m_LiDARFrame = frame;
	//m_Panoramas.push_back(QImage(panorama, m_width, m_height, QImage::Format::Format_RGB888));
	m_lidarInputCount++;
	lidarFrameMutex.unlock();
}

QMap<int, void*> StreamProcess::getAudioFrameSeq()
{
	QMap<int, void*> audioFrameSeq;
	/*audioFrameMutex.lock();
	int readyChannelCount = 0;
	for (int i = 0; i < m_audioChannelCount; i++)
	{
		if (m_audioFrames[i].size() >= 1)
		{
			readyChannelCount++;
		}
	}
	if (readyChannelCount == m_audioChannelCount)
	{
		for (int i = 0; i < m_audioChannelCount; i++)
		{
			audioFrameSeq[i] = m_audioFrames[i][0];
			m_audioFrames[i].pop_front();
		}
		m_audioInputCount++;
	}
	audioFrameMutex.unlock();*/
	return audioFrameSeq;
}

bool StreamProcess::isFinished()
{
	return m_finished;	
}

void StreamProcess::playAndPause(bool isPause)
{
	QMutexLocker locker(&doPauseMutex);
	doPause = isPause;
}

//
// this is Process WebRTC threads
//

webRTC_StreamProcess::webRTC_StreamProcess(SharedImageBuffer *sharedImageBuffer, bool toFile, QObject* main)
	:StreamProcess(sharedImageBuffer, toFile, main)
{
	m_Name = "webRTCStreamProcess";
	m_iswebRTC = true;
}

bool webRTC_StreamProcess::initializeStream(int nSerialNumber)
{
	if (nSerialNumber == 0)
		init(); // should reset again if (splitMins > 0.f)

	m_nCurrentSplitSerialNumber = nSerialNumber;

	int ret = m_Stream.Initialize_webRTC(
		m_outFileName.toUtf8().constData(), m_width, m_height, m_fps, 
		m_channelCount, m_sampleFmt, m_srcSampleRate, m_sampleRate, m_audioLag, 
		m_toFile, m_videoCodec, m_audioCodec, m_crf);

	if (ret >= 0)
	{
		if (m_sharedImageBuffer->getGlobalAnimSettings()->m_captureType == D360::Capture::CaptureDomain::CAPTURE_DSHOW)
			m_Stream.setCaptureType(D360::Capture::CaptureDomain::CAPTURE_DSHOW);

		// Get expected audio channel count for output video
		m_audioChannelCount = m_sharedImageBuffer->getGlobalAnimSettings()->getAudioChannelCount();

		if (streamingThreadInstance == NULL)
		{
			streamingThreadInstance = new QThread();
			connect(streamingThreadInstance, SIGNAL(started()), this, SLOT(process()));
			if (m_Main) {
				connect(this, SIGNAL(finished(int, QString, int)), m_Main, SLOT(finishedThread(int, QString, int))/*, Qt::DirectConnection*/);
				connect(this, SIGNAL(started(int, QString, int)), m_Main, SLOT(startedThread(int, QString, int)));
			}
		}
		if (streamingThreadInstance != NULL)
		{
			this->moveToThread(streamingThreadInstance);
			streamingThreadInstance->start();
		}

		m_isOpened = true;
		return true;
	}
	return false;
}

