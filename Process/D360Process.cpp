#include <QApplication>
#include <QThreadPool>
#include <QFile>

#include "D360Process.h"
#include "QmlMainWindow.h"

#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

extern QmlMainWindow* g_mainWindow;

D360Process::D360Process()
{
	//
	// Create SharedImageBuffer object
	//
	m_sharedImageBuffer = new SharedImageBuffer();

	udpSocket = NULL;

#if defined FAST_VIDEO
	FastVideoInit();
#endif

	stitcherThread = std::shared_ptr<D360Stitcher>(new D360Stitcher(m_sharedImageBuffer));
	playbackStitcherThread = std::shared_ptr<PlaybackStitcher>(new PlaybackStitcher(m_sharedImageBuffer));

	udpSocket = new QUdpSocket(this);
	udpSocket->bind(QHostAddress::Any, 0, QUdpSocket::DefaultForPlatform);
	//connect( udpSocket, SIGNAL( readyRead() ), this, SLOT( readPendingDatagrams() ) );
}

D360Process::~D360Process()
{
	stitcherThread = NULL;
	playbackStitcherThread = NULL;
	if (m_sharedImageBuffer)
	{
		delete m_sharedImageBuffer;
		m_sharedImageBuffer = NULL;
	}

	delete udpSocket;
}

void D360Process::initialize()
{
	m_sharedImageBuffer->initialize();
}

void D360Process::readPendingDatagrams()
{
	while (udpSocket->hasPendingDatagrams())
	{
		QByteArray datagram;
		datagram.resize(udpSocket->pendingDatagramSize());
		QHostAddress sender;
		quint16 senderPort;

		udpSocket->readDatagram(datagram.data(), datagram.size(), &sender, &senderPort);

		processTheDatagram(datagram);
	}
}

void D360Process::processTheDatagram(QByteArray& datagram)
{
	QString qs(datagram);
}

bool D360Process::connectPlaybackThread()
{
	QImage captureImage(g_mainWindow->getCaptureImageFilePath());
	playbackThread = new PlaybackThread(m_sharedImageBuffer, D360::Capture::CaptureDomain::CAPTURE_VIDEO, captureImage.size().width(), captureImage.size().height());
	return playbackThread->connect();
}

void D360Process::startPlaybackThread()
{
	QThread* playbackThreadInstance = new QThread;
	playbackThread->moveToThread(playbackThreadInstance);

	connect(playbackThreadInstance, SIGNAL(started()), playbackThread, SLOT(process()));
	connect(playbackThreadInstance, SIGNAL(finished()), playbackThread, SLOT(deleteLater()));

	playbackThreadInstance->start();
}

bool D360Process::attach(CameraInput& camSettings, int deviceNumber, float startframe, float endframe, float fps, int imageBufferSize, int nextTabIndex)
{
	std::cout << "Loading Camera: " << camSettings.name.toStdString() << std::endl;

	// Check if this camera is already connected
	//
	if (!m_deviceNumberMap.contains(deviceNumber))
	{
		//
		// Create ImageBuffer with user-defined size
		//
		//Buffer< cv::Mat > *imageBuffer = new Buffer< cv::Mat >( imageBufferSize );

		//
		// Add created ImageBuffer to SharedImageBuffer object
		//
		m_sharedImageBuffer->add(deviceNumber, false);
		GlobalState s;
		s.m_curFrame = startframe;

		m_sharedImageBuffer->addState(deviceNumber, s, false);
		m_sharedImageBuffer->setGlobalAnimSettings(&m_d360Data.getGlobalAnimSettings());
		//
		// Create CameraView
		//
		//std::cout << "New Cam View" << std::endl;
		//m_cameraViewMap[ deviceNumber ] = new CameraView( NULL, m_sharedImageBuffer );

		std::cout << "Connecting Cam" << std::endl;

		//
		// Create capture thread
		//
		if (deviceNumber >= D360_FILEDEVICESTART)
			captureThread = new CaptureThread(m_sharedImageBuffer, deviceNumber, D360::Capture::CaptureDomain::CAPTURE_FILE, camSettings.xres, camSettings.yres);
		else
			captureThread = new CaptureThread(m_sharedImageBuffer, deviceNumber, D360::Capture::CaptureDomain::CAPTURE_DSHOW, camSettings.xres, camSettings.yres);

		captureThread->setGain(camSettings.exposure);
		captureThread->setExposure(1.0f);
		captureThread->setFrameRate(camSettings.fps);

		//
		// Attempt to connect to camera
		if (captureThread->connect())
		{
#if 0 /*[C]*/			 
			// Create processing thread
			processingThread = new ProcessingThread(m_sharedImageBuffer, deviceNumber);
#endif
			//
			// Start capturing frames from camera
			//
			//captureThread->start( QThread::Priority::TimeCriticalPriority );
			// Start processing captured frames (if enabled)
			//processingThread->start( QThread::Priority::TimeCriticalPriority );

			//
			// Add to map
			//
			m_deviceNumberMap[deviceNumber] = nextTabIndex;
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}

bool D360Process::connectToCamera()
{
	GlobalAnimSettings::CameraSettingsList& cameraSettingsList = m_d360Data.getGlobalAnimSettings().cameraSettingsList();
	float startframe = m_d360Data.startFrame();
	float endframe = m_d360Data.endFrame();
	float fps = 60.0f;

	if (cameraSettingsList.size() > 0)
	{
		for (unsigned i = 0; i < cameraSettingsList.size(); ++i)
		{
			int deviceNumber = i;
			CameraInput& camInput = cameraSettingsList[i];
			bool retval = attach(camInput, deviceNumber, startframe, endframe, fps, i + 1);
			return retval;
		}
	}
	return true;
}

void D360Process::connectToCameras()
{
	// Get next tab index
	GlobalAnimSettings::CameraSettingsList& cameraSettingsList = m_d360Data.getGlobalAnimSettings().cameraSettingsList();

	int startframe = m_d360Data.getGlobalAnimSettings().m_startFrame;
	int endframe = m_d360Data.getGlobalAnimSettings().m_endFrame;
	float fps = m_d360Data.getGlobalAnimSettings().m_fps;

	if (cameraSettingsList.size() > 0)
	{
		for (unsigned i = 0; i < cameraSettingsList.size(); ++i)
		{
			int deviceNumber = i;
			int nextTabIndex = i;
			CameraInput& camInput = cameraSettingsList[i];

			attach(camInput, deviceNumber, startframe, endframe, fps, nextTabIndex);
		}
	}
}

void D360Process::loadSavedFrames()
{
	GlobalAnimSettings::CameraSettingsList& cameraSettingsList = m_d360Data.getGlobalAnimSettings().cameraSettingsList();

	float startframe = m_d360Data.startFrame();
	float endframe = m_d360Data.endFrame();
	float fps = 60.0f;

	int imageBufferSize = fps * (endframe - startframe);
	int lastDeviceNumber = -1;
	//
	// Save user-defined device number
	//
	std::cout << "Number of Cameras: " << cameraSettingsList.size() << std::endl;
	for (unsigned i = 0; i < cameraSettingsList.size(); ++i)
	{
		int deviceNumber = i + D360_FILEDEVICESTART;
		CameraInput& camInput = cameraSettingsList[i];
		attach(camInput, deviceNumber, startframe, endframe, fps, i - D360_FILEDEVICESTART);
	}
}

void D360Process::disconnectCameras()
{
	GlobalAnimSettings::CameraSettingsList& cameraSettingsList = m_d360Data.getGlobalAnimSettings().cameraSettingsList();

	for (unsigned i = 0; i < cameraSettingsList.size(); ++i)
	{
		disconnectCamera(i);
	}
}

void D360Process::disconnectCamera(int index)
{
#if 0 /*[C]*/
	delete m_cameraViewMap[m_deviceNumberMap.key(index)];
	m_cameraViewMap.remove(m_deviceNumberMap.key(index));
#endif
	// Remove from map
	removeFromMapByTabIndex(m_deviceNumberMap, index);
}

bool D360Process::removeFromMapByTabIndex(QMap<int, int> &map, int tabIndex)
{
	QMutableMapIterator<int, int> i(map);
	while (i.hasNext())
	{
		i.next();
		if (i.value() == tabIndex)
		{
			i.remove();
			return true;
		}
	}

	return false;
}

void D360Process::record()
{
}

void D360Process::pause()
{
}

void D360Process::captureNew()
{
	disconnectCameras();
	m_d360Data.clear();
}

void D360Process::close()
{
#if 0 /*[commented by B]*/
	std::cout << "Terminating threads" << std::endl;
#if 0 /*[C]*/
	for (QMap<int, CameraView *>::const_iterator itr = m_cameraViewMap.begin(); itr != m_cameraViewMap.end(); ++itr)
	{
		std::cout << "Deleting Views" << std::endl;
		delete *itr;
	}
#endif
	QApplication::quit();
#endif
}