#include "scy/idler.h"
#include "scy/socketio/client.h"
#include "signaler.h"

#include "webrtc/base/ssladapter.h"
#include "webrtc/base/thread.h"

#include "QmlMainWindow.h"
#include <QApplication>
#include "SharedImageBuffer.h"
#include "QmlRecentDialog.h"
#include "D360Parser.h"
#include "define.h"
#include "SlotInfo.h"
#include <QDir>
#include <QtXml/QDomDocument>
#include <QQuickWindow>
#include <iostream>
#include <sstream>
#include <QFile>
#include <QScreen>
#include <QDesktopWidget>
#include <QtMath>
#include <QTreeView>
#include "3DMath.h"
#include "include/Config.h"
#include "ConfigZip.h"
#include "LiDARThread.h"

QmlMainWindow* g_mainWindow = NULL;
int snapshotCount = 0;
//PanoLog* g_logger = NULL;

//int BannerInfo::seedId = 0;
TakeMgrTreeModel* g_takeMgrModel = NULL;
extern CoreEngine* g_Engine;
// QString convertNumber2String( int number, int minLength )
// {
// 	QString strNumber = QString::number( number );
// 	int diff = minLength - strNumber.length();
// 	if( diff <= 0 )
// 		return strNumber;
// 	QString prefix = "";
// 	for( int i = 0; i < diff; i++ )
// 		prefix += "0";
// 	return QString( "%1%2" ).arg( prefix ).arg( strNumber );
// }

#define FAVORITE_XML_FILENAME				"favorite.xml"
#define TEMPLATE_FOLDERNAME					"template"

#define generateSessionName(sessionId)				QString("%1_%2").arg(QDate::currentDate().toString("yyyy-MM-dd")).arg(convertNumber2String(sessionId, 2))
#define generateSessionDirFullPath(sessionId)		QString("%1%2").arg(m_sessionRootPath).arg(generateSessionName(sessionId))
#define generateSessionLogFileFullPath(sessionId)	QString("%1/session.log").arg(generateSessionDirFullPath(sessionId))
#define generateSessionDirFullpathFromData(str)		QString("%1%2").arg(m_sessionRootPath).arg(str)	
#define generateSessionLogFileFullPathFromData(str)	QString("%1/session.log").arg(generateSessionDirFullpathFromData(str))

QmlMainWindow::QmlMainWindow()
: m_Name("Look3D")
, m_isExit(false)
, m_isStarted(false)
, m_calibResult(false)
, m_oculusDevice(NULL)
, m_isEndCaptureThreads(true)
{
	g_mainWindow = this;
	m_viewMode = LIVE_VIEW;		
	
	m_streamProcess = 0;
	m_process = 0;
	sharedImageBuffer = 0;
	
	m_stitcherView = 0;
	m_stitcherFullScreenView = 0;
	m_interactView = NULL;

	m_playbackModule = NULL;
	m_selfCalib = NULL;

	m_eTimeValue = "";
	m_fpsValue = "";
	m_nSliderValue = 0;

	m_connectedCameras = false;
	m_reconnectCameras = false;
	m_isPlayFinished = false;
	m_isStopPlayback = false;
	m_isClose = false;	

	m_isDisconnectComplete = true;

	m_isAboutToStop = true;

	if (m_process == NULL) {		// Only will be call once.
		m_process = new D360Process;
		sharedImageBuffer = m_process->getSharedImageBuffer();
		sharedImageBuffer->setStitcher(m_process->getStitcherThread());
		sharedImageBuffer->setPlaybackStitcher(m_process->getPlaybackStitcherThread());
		sharedImageBuffer->setGlobalAnimSettings(&m_d360Data.getGlobalAnimSettings());

		//PANO_LOG(">Linking signal channel on stitcher ...");
		// Process initialize
		{
			qRegisterMetaType< struct ThreadStatisticsData >("ThreadStatisticsData");
			PlaybackStitcher* pPlaybackStitcher = sharedImageBuffer->getPlaybackStitcher().get();
			connect(pPlaybackStitcher, SIGNAL(updateStatisticsInGUI(struct ThreadStatisticsData)),
				this, SLOT(updatePlaybackStitchingThreadStats(struct ThreadStatisticsData)), Qt::DirectConnection);

			D360Stitcher* pStitcher = sharedImageBuffer->getStitcher().get();

			connect(pStitcher, SIGNAL(newPanoramaFrameReady(unsigned char*)),
				this, SLOT(streamPanorama(unsigned char*)), Qt::DirectConnection);

			connect(pStitcher, SIGNAL(updateStatisticsInGUI(struct ThreadStatisticsData)),
				this, SLOT(updateStitchingThreadStats(struct ThreadStatisticsData)), Qt::DirectConnection);

			connect(pStitcher, SIGNAL(finished(int, QString, int)), this, SLOT(finishedThread(int, QString, int)));
			connect(pStitcher, SIGNAL(started(int, QString, int)), this, SLOT(startedThread(int, QString, int)));

			connect(pStitcher, SIGNAL(snapshoted()), this, SLOT(finishedSphericalSnapshot()));
		}
	}

	// 	m_blMousePressed = false;
	initDevices();

	emit started(false);			// Sending started signal to QML.		

	m_calib = new CalibProcess("", this);

	m_logger.setParent(this);
	m_logger.initialize(PANO_LOG_LEVEL::WARNING, "PanoOneHistory.log");
	m_logger.enableStdout();
	g_logger = &m_logger;

	m_deletableCameraCnt = 0;
	
	m_weightMap_cameraIndex = 0;
	m_weightMap_radius = 0;
	m_weightMap_strength = 0;
	m_weightMap_fallOff = 0;
	m_weightMap_isIncrement = true;
	m_weightMap_eyeMode = WeightMapEyeMode::DEFAULT;
	m_eyeMode = WeightMapEyeMode::DEFAULT;
	m_weightMapEditCameraIndex = 0;
	m_weightMapEditUndoStatus = false;
	m_weightMapEditRedoStatus = false;

	// Take management
	m_sessionRootPath = "C:/Capture/";
	m_lastFramesProcessed = 0;
	m_isNewConfiguration = true;

	m_playMode = START_MODE;
	m_playbackMode = START_MODE;
	m_qmlTakeManagement = NULL;
	m_sessionNameList = {};
	m_takeNameList = {};
	m_templateIniFile = "";	
}


QmlMainWindow::~QmlMainWindow()
{
	//closeMainWindow();
	if (m_process)
	{
		delete m_process;
		m_process = NULL;
	}

	delete m_calib;
	{
		g_mainWindow = NULL;
		QGuiApplication::quit();
	}
}

void QmlMainWindow::releaseThreads()
{
	if (m_streamProcess && m_streamProcess->isWebRTC() == false)
	{
		delete m_streamProcess;
		m_streamProcess = NULL;
	}

	if (m_oculusDevice)
	{
		delete m_oculusDevice;
		m_oculusDevice = NULL;
	}

	for (int i = 0; i < m_bannerThreads.size(); i++)
	{
		BannerThread* banner = m_bannerThreads[i];
		delete banner;
		banner = NULL;
	}		
	m_bannerThreads.clear();

#if 0 /*[C]*/
	if (liveView)
	{
		delete liveView;
		liveView = 0;
	}
	if (stitcherView)
	{
		delete stitcherView;
		stitcherView = 0;
	}
#endif

	if (m_selfCalib)
	{
		delete m_selfCalib;
		m_selfCalib = NULL;
	}

	//deviceNumberMap.clear();
}

void QmlMainWindow::receiveKeyIndex(int level)
{
}

int QmlMainWindow::level()
{ 
	if (getGlobalAnimSetting().m_blendingMode == GlobalAnimSettings::Feathering)
		return -1;
	else if (getGlobalAnimSetting().m_blendingMode == GlobalAnimSettings::MultiBandBlending)
	{
		return getGlobalAnimSetting().m_multiBandLevel;
	}
	return 0;
}

void QmlMainWindow::closeMainWindow()
{	
	if (m_playMode == PAUSE_MODE)
	{
		// if it is playing now, pause first, then call disconnectCameras();
		setPlayMode(START_MODE);
	}

	if (m_playbackMode == PAUSE_MODE)
	{
		// if it is playing now, pause first, then call disconnectPlayback();
		setPlaybackMode(START_MODE);
	}

	//QThread::msleep(300);	

	m_isExit = true;
	saveRecentMgrToINI();
	
	if (m_connectedCameras) {
		disconnectPlayback();
		disconnectCameras();
	} else {
		finishedThread(THREAD_TYPE_MAIN);
	}
}

extern QString getStringFromWString(wstring wstr);

void QmlMainWindow::reportError(int type, QString msg, int id) {
	QString typeStr = "";
	static int cameraThreadsEndNum = 0;
	switch (type)
	{
	case THREAD_TYPE_MAIN:
		typeStr = m_Name;
		break;
	case THREAD_TYPE_STITCHER:
		typeStr = "Stitcher";
		break;
	case THREAD_TYPE_CAPTURE:
		typeStr = "Capture";
		if (msg == "EOF")
			cameraThreadsEndNum++;
		break;
	case THREAD_TYPE_AUDIO:
		typeStr = "Capture";
		break;
	case THREAD_TYPE_LIDAR:
		typeStr = "LiDAR";
		break;
	case THREAD_TYPE_STREAM:
		typeStr = "Streaming";
		break;
	case THREAD_TYPE_OCULUS:
		typeStr = "Oculus";
		break;
	case THREAD_TYPE_PLAYBACK:
		typeStr = "Playback";
		break;
	default:
		typeStr = m_Name;
		break;
	}

	QString errorStr = QString("[%1] [%2] [ERROR] %3").arg(typeStr).ARGN(id).arg(msg);
		
	if (cameraThreadsEndNum == m_d360Data.getGlobalAnimSettings().cameraSettingsList().size()) 
	{
		PANO_ERROR(errorStr);
		m_isEndCaptureThreads = true;
	}
	else 
	{
		if (msg != "EOF") {			
			PANO_N_ERROR(errorStr);
		}		
		//emit setErrorMsg(errorStr);
	}
}

void QmlMainWindow::startedThread(int type, QString msg, int id)
{
	QString typeStr = "";
	static int cameraThreadsStartedNum = 0;
	static int audioThreadsStartedNum = 0;
	switch (type)
	{
	case THREAD_TYPE_MAIN:
		typeStr = m_Name;
		break;
	case THREAD_TYPE_CAPTURE:
		typeStr = "Capture";
		cameraThreadsStartedNum++;
		break;
	case THREAD_TYPE_BANNER:
		typeStr = "Banner";
		break;
	case THREAD_TYPE_AUDIO:
		typeStr = "Audio";
		audioThreadsStartedNum++;
		break;
	case THREAD_TYPE_LIDAR:
		typeStr = "LiDAR";
		break;
	case THREAD_TYPE_STITCHER:
		typeStr = "Stitcher";
		// Camera threads start...
		if (msg == "0" || msg == "")
		{
			// This is the first of split video output, or single mode.
			for (unsigned i = 0; i < cameraModuleMap.size(); ++i)
			{
				if (cameraModuleMap[i]->isConnected())
					cameraModuleMap[i]->startThreads();		// Must start after stream thread started!
			}

			for (unsigned i = 0; i < m_nodalCameraModuleMap.keys().size(); i++)
			{
				if (m_nodalCameraModuleMap.values()[i]->isConnected())
					m_nodalCameraModuleMap.values()[i]->startThreads();
			}

			if (m_d360Data.getGlobalAnimSettings().m_oculus)
				break;

			// Audio threads start...
			for (unsigned i = 0; i < cameraModuleMap.size(); ++i)
			{
				if (m_d360Data.getGlobalAnimSettings().getCameraInput(i).isExistAudio() == false)
					continue;

				AudioThread* audioThread = audioModuleMap[i];
				AudioInput* mic = audioThread->getMic();
				if (mic)	audioThread->startThread();
			}
		}
		break;
	case THREAD_TYPE_STREAM:
		typeStr = "Streaming";
		break;
	case THREAD_TYPE_OCULUS:
		typeStr = "Oculus";
		break;
	case THREAD_TYPE_PLAYBACK:
		typeStr = "Playback";
		break;
	default:
		typeStr = m_Name;
		break;
	}

	PANO_LOG(QString("[%1] [%2] [Started] %3").arg(typeStr).ARGN(id).arg(msg));

	if (cameraThreadsStartedNum == m_d360Data.getGlobalAnimSettings().cameraSettingsList().size() &&
		audioThreadsStartedNum == getAudioSelCnt())
	{
		cameraThreadsStartedNum = 0;
		audioThreadsStartedNum = 0;
	}
}

void QmlMainWindow::finishedThread(int type, QString msg, int id)
{
	QString typeStr = "";
	static int cameraThreadsFinishedNum = 0;
	static int audioThreadsFinishedNum = 0;
	static int cameraThreadsEndNum = 0;
	bool isFinal = false;
	switch (type)
	{
	case THREAD_TYPE_MAIN:
		typeStr = m_Name;
		if (m_isExit == true) {
			setExit(true);
		}
		return;
	case THREAD_TYPE_STITCHER:
		typeStr = "Stitcher";
		if (m_streamProcess)
		{
			if (!m_streamProcess->isFinished())
			{
				if (m_streamProcess->isTakeRecording())
					stopRecordTake(NULL);
				else
					stopStreaming();
			}
		}		
		{
			bool isOculusRunning = m_oculusDevice && !m_oculusDevice->isFinished();
			if (isOculusRunning)
			{
				m_oculusDevice->stop(OculusRender::StopReason::FINISH_CONFIGURATION);
			}

			isFinal = true;
		}
		if (m_process->getStitcherThread())
			m_process->getStitcherThread()->setFinished();
		break;
	case THREAD_TYPE_CAPTURE:
		typeStr = "Capture";
		sharedImageBuffer->setCaptureFinalizing();
		if (msg == "EOF")
			cameraThreadsEndNum++;
		else
			cameraThreadsFinishedNum++;
		break;
	case THREAD_TYPE_BANNER:
		typeStr = "Banner";
		for (int i = 0; i < m_bannerThreads.size(); i++)
		{
			BannerThread* banner = m_bannerThreads[i];
			if (banner->getBannerId() == id)
			{
				banner->setFinished();				
				break;
			}
		}
		break;
	case THREAD_TYPE_AUDIO:
		typeStr = "Audio";
		audioThreadsFinishedNum++;
		break;
	case THREAD_TYPE_LIDAR:
		typeStr = "LiDAR";
		break;
	case THREAD_TYPE_OCULUS:
		typeStr = "Oculus";
		if (msg == MSG_OCULUS_SWITCHOFF)
		{
			/*if (m_oculusDevice)
			{
				delete m_oculusDevice;
				m_oculusDevice = NULL;
			}*/
		}
		else
			isFinal = true;

		if (m_oculusDevice)
			m_oculusDevice->setFinished();
		break;
	case THREAD_TYPE_STREAM:
		typeStr = "Streaming";

		if (m_streamProcess)
		{
			if (m_streamProcess->isWebRTC() == false)
			{
				delete m_streamProcess;
				m_streamProcess = NULL;
			}
			sharedImageBuffer->setStreamer(NULL);
			emit loadingMessageReceived("StreamingFailed");
		}
		break;
	case THREAD_TYPE_PLAYBACK:
		typeStr = "Playback";
		break;
	default:
		typeStr = m_Name;
		break;
	}

	PANO_LOG(QString("[%1] [%2] [Finished] %3 : %4").arg(typeStr).ARGN(id).arg(m_isExit ? " [Exit]" : "").arg(msg));

	//qDebug() << "finished status: " << cameraThreadsEndNum << cameraThreadsEndNum << m_deletableCameraCnt << m_isStarted;

	// for auto-Replay function
	if (cameraThreadsEndNum > 0 && cameraThreadsEndNum == m_deletableCameraCnt && m_isStarted) {
		PANO_N_LOG(QString("[%1] [All videos finished to play]").arg(typeStr));
		cameraThreadsEndNum = 0;
		setPlayFinish(true); // Sending playFinish signal to QML.			
		streamClose();
	}

	//Only must be stop stitcher thread after stop all thread associated with cameras.
	bool bannerThreadFinished = true;
	for (int i = 0; i < m_bannerThreads.size(); i++)
	{
		bannerThreadFinished = bannerThreadFinished && m_bannerThreads[i]->isFinished();
	}
	
	PANO_LOG(QString("cameraEndNum: %1, cameraFinishedNum: %2, cameraCount: %3, started: %4, bannerFinished:%5, isFinal:%6").
		arg(cameraThreadsEndNum).arg(cameraThreadsFinishedNum).arg(m_deletableCameraCnt).arg(m_isStarted).arg(bannerThreadFinished).arg(isFinal));

	if (cameraThreadsEndNum + cameraThreadsFinishedNum == m_deletableCameraCnt &&
		audioThreadsFinishedNum == m_deletableAudioCnt &&
		bannerThreadFinished && m_isStarted)
	{

		PANO_LOG("It's turn to stop stitcher thread");

		//Only must be stop stitcher thread after stop all thread associated with cameras.		
		if (!m_process->getStitcherThread()->isFinished())
		{
			cameraThreadsFinishedNum = 0;
			audioThreadsFinishedNum = 0;

			emit loadingMessageReceived("Stopping stitcher engine...");
			m_process->getStitcherThread()->stopStitcherThread();
		}
		m_isStarted = false;
		//emit started(false);			// Sending started signal to QML.				
	}

	if (isFinal)
	{
		cameraThreadsFinishedNum = 0;
		audioThreadsFinishedNum = 0;
		cameraThreadsEndNum = 0;

		releaseThreads();
		PANO_N_LOG(QString("[Look3D] Finished all threads"));

		m_process->initialize();

		setDisconnectComplete(true);

		if (m_isExit == true) {
			setExit(true);		// On this function, send onExitChagned signal
		}

		/*if (!m_isExit && m_connectedCameras) {*/
		if (!m_isExit && !m_isClose) {
			m_connectedCameras = false;					
			connectToCameras();			
		}
	}
}

void QmlMainWindow::initDevices()
{

	CaptureDevices *capDev = new CaptureDevices();
    std::vector<DeviceMetaInformation> videoDevices;
    std::vector<std::wstring> audioDevicesW;
    m_videoDevices.clear();
    capDev->GetVideoDevices(&videoDevices);
	capDev->GetAudioDevices(&audioDevicesW);
	delete capDev;
	m_cameraCnt = videoDevices.size();
    m_audioCnt = audioDevicesW.size();
	for (int i = 0; i < videoDevices.size(); i++) {
		m_videoDevices[i] = videoDevices[i];
		//m_audioDevices[i] = getStringFromWString(audioDevicesW[i]);
	}
		
		
}

bool QmlMainWindow::isSelectedCamera(QString name)
{
	for (unsigned i = 0; i < m_d360Data.getGlobalAnimSettings().cameraSettingsList().size(); i++)
	{
		if (m_d360Data.getGlobalAnimSettings().getCameraInput(i).name == name)
			return true;
	}

	// check nodal camera name
	GlobalAnimSettings& gasettings = m_d360Data.getGlobalAnimSettings();	
	if (gasettings.isNodalAvailable())
	{
		CameraInput& nodalCameraInput = gasettings.getCameraInput(NODAL_CAMERA_INDEX);

		if (nodalCameraInput.name == name)
			return true;
	}
	return false;
}

bool QmlMainWindow::isSelectedAudio(QString name)
{
	for (unsigned i = 0; i < m_d360Data.getGlobalAnimSettings().cameraSettingsList().size(); i++)
	{
		if (m_d360Data.getGlobalAnimSettings().getCameraInput(i).audioName == name)
			return true;
	}
	return false;
}

int QmlMainWindow::getAudioSelCnt()
{
	int count = 0;
	for (unsigned i = 0; i < m_d360Data.getGlobalAnimSettings().cameraSettingsList().size(); i++)
	{
		if (m_d360Data.getGlobalAnimSettings().getCameraInput(i).isExistAudio() == false)
			continue;

		count++;
	}

	return count;
}

void QmlMainWindow::resetConfigList()
{
	QMap<int, QString> list;
	m_d360Data.clearStereoList();
	for (unsigned i = 0; i < m_d360Data.getGlobalAnimSettings().cameraSettingsList().size(); i++)
	{
		if (m_d360Data.getGlobalAnimSettings().m_captureType == D360::Capture::CaptureDomain::CAPTURE_FILE)
		{
			list[i] = m_d360Data.getGlobalAnimSettings().getCameraInput(i).fileDir;
			m_d360Data.setTempImagePrefix(i, m_d360Data.getGlobalAnimSettings().getCameraInput(i).filePrefix);
		}			
		else
			list[i] = m_d360Data.getGlobalAnimSettings().getCameraInput(i).name;
		m_d360Data.setTempStereoType(i, m_d360Data.getGlobalAnimSettings().getCameraInput(i).stereoType);
		if (m_d360Data.getGlobalAnimSettings().m_captureType != D360::Capture::CAPTURE_DSHOW)
			m_d360Data.setTempAudioSettings(i, m_d360Data.getGlobalAnimSettings().getCameraInput(i).audioType);
	}
	switch (m_d360Data.getGlobalAnimSettings().m_captureType)
	{
	case D360::Capture::CaptureDomain::CAPTURE_VIDEO:
		m_videoPathList.clear(); m_videoPathList = list; break;
	case D360::Capture::CaptureDomain::CAPTURE_FILE:
		m_imagePathList.clear(); m_imagePathList = list; break;
	default:	// CAPTURE_DSHOW, CAPTURE_XIMIA, CAPTURE_OPENCV
		break;
	}

	m_d360Data.getGlobalAnimSettings().m_fYaw = 0.f;
	m_d360Data.getGlobalAnimSettings().m_fPitch = 0.f;
	m_d360Data.getGlobalAnimSettings().m_fRoll = 0.f;

	m_d360Data.resetTempGlobalAnimSettings();
}

void QmlMainWindow::connectToCameras()
{
	emit loadingMessageReceived("Loading...");

	PANO_LOG(">Start connecting to cameras ...");

	resetConfigList();	// For after reconfiguration.
	snapshotCount = 0;

	// Multi-Nodal
	m_nodalCameraModuleMap.clear();

	GlobalAnimSettings& gasettings = m_d360Data.getGlobalAnimSettings();

	QMap<int, CameraInput> nodalCameraMap; nodalCameraMap.clear();
	bool isNodal = gasettings.isNodalAvailable();
	bool isNodalCameraAvailable = false;
	bool isNodalOffline = false;
	if (isNodal)
	{
		bool isNodalCameraAvailable = gasettings.detachNodalCamera(nodalCameraMap);
		if (isNodalCameraAvailable)
			m_d360Data.setCameraCount(m_d360Data.getCameraCount() - 1);
		else
		{
			gasettings.getOfflineCamera(nodalCameraMap);
		}
	}

	GlobalAnimSettings::CameraSettingsList& cameraSettingsList = gasettings.cameraSettingsList();

	int panoWidth = gasettings.m_panoXRes;
	int panoHeight = gasettings.m_panoYRes;

	int startframe = gasettings.m_startFrame;
	int endframe = gasettings.m_endFrame;
	float fps = gasettings.m_fps;
	int nCameras = cameraSettingsList.size();

	gasettings.useCuda = m_qmlApplicationSetting->useCUDA();

	std::shared_ptr< D360Stitcher > stitcher = m_process->getStitcherThread();

	sharedImageBuffer->setStitcher(stitcher);
	sharedImageBuffer->setGlobalAnimSettings(&gasettings);

	if( gasettings.getLiDARPort() != -1 )
	{
		PANO_LOG( ">Creating LiDAR threads ..." );
		LiDARThread* lidarThread = new LiDARThread( this );
		lidarThread->initialize( sharedImageBuffer );
		LiDARInput* cap = lidarThread->getInputDevice();
		if( cap )
			lidarThread->startThread();
	}

	if (nCameras <= 0) {
		PANO_ERROR(QString("Number of camera is invalid! (%1)").ARGN(nCameras));
		return;
	}

	PANO_LOG(">Creating audio threads ...");
	//if (!gasettings.m_oculus)
	{
		for (unsigned i = 0; i < nCameras; ++i)
		{
			if (gasettings.getCameraInput(i).isExistAudio() == false)
				continue;

			QString("Initializing audio engine #%1...").ARGN(i);

			AudioThread* audioThread = new AudioThread(this);
			audioThread->initialize(sharedImageBuffer, i);
#if 0	// Capture threads will be started after stream/stitching threads started!
			AudioInput* mic = audioThread->getMic();
			if (mic)	audioThread->startThread();
#endif
			audioModuleMap[i] = audioThread;
		}
	}

	PANO_LOG(">Creating cameras ...");
	for (unsigned i = 0; i < cameraSettingsList.size(); ++i)
	{
		PANO_LOG(QString("Attaching camera(%1)...").ARGN(i));

		emit loadingMessageReceived(QString("Connecting to camera #%1...").ARGN(i));

		CameraInput& camInput = cameraSettingsList[i];
		if (!attach(camInput, i, startframe, endframe, fps)) {
			PANO_N_ERROR(QString("Camera(%1) connecting failed!").ARGN(i));
			continue;
		}

		PANO_LOG(QString("Attached camera(%1).").ARGN(i));
	}

	if (isNodal)
	{
		for (unsigned i = 0; i < nodalCameraMap.keys().size(); i++)
		{
			PANO_LOG(QString("Attaching nodal camera..."));
			emit loadingMessageReceived(QString("Connecting to nodal camera..."));

			if (!attachNodalCamera(nodalCameraMap.values()[i],i, nCameras))
			{
				PANO_N_ERROR(QString("Nodal camera connecting failed!"));
			}
			else
			{
				PANO_LOG(QString("Attached nodal camera."));
			}
		}
	}

	m_deletableCameraCnt = cameraModuleMap.size();
	m_deletableAudioCnt = audioModuleMap.size();

	if (isNodalCameraAvailable)
		m_deletableCameraCnt++;

	PANO_LOG(">Initializing stitcher ...");
	emit loadingMessageReceived("Initializing stitching engine...");
	stitcher->init(QOpenGLContext::globalShareContext());
	PANO_LOG(">Starting stitcher thread ...");
	emit loadingMessageReceived("Starting stitcher...");
	stitcher->startThread();

	PANO_LOG(">Creating stream instances ...");
	emit loadingMessageReceived("Connecting to stream instance (this will take a few seconds)...");
	int channels = gasettings.getAudioChannelCount();
	int resultHeight = panoHeight;
	if (sharedImageBuffer->getGlobalAnimSettings()->isStereo())
		resultHeight *= 2;

	bool blLiveMode = gasettings.m_captureType == D360::Capture::CAPTURE_DSHOW;
	
#if 0		// Capture threads will be started after stream/stitching threads started!
	for (unsigned i = 0; i < cameraModuleMap.size(); ++i)
	{
		if (cameraModuleMap[i]->isConnected())
			cameraModuleMap[i]->startThreads();
	}
#endif
	emit loadingMessageReceived("Configuration loading complete...");

	PANO_LOG(">Loading banner instances ...");
	GlobalAnimSettings::BannerArray& banners = gasettings.m_banners;
	BannerInfo::seedId = 0;
	for (int i = 0; i < banners.size(); i++)
	{
		BannerInput& banner = banners[i];
		addBanner(banner.quad, banner.bannerFile, banner.isVideo, banner.isStereoRight);
	}

	if (blLiveMode)
	{
		// Set live camera settings for configuration template
		initTemplateCameraObject();
		for (unsigned i = 0; i < cameraSettingsList.size(); ++i)
		{
			//sendCameraName(i, cameraSettingsList[i].name, (i == gasettings.m_nodalVideoIndex), gasettings.m_hasNodalOfflineVideo);
			sendCameraName(i, cameraSettingsList[i].name, false, false);
		}
	}

	// Clear sessionTake variables.
	m_isNewConfiguration = true;
	if (m_sessionNameList.size() != 0) {
		g_takeMgrModel->removeRows(0, m_sessionNameList.size());
	}
	m_sessionNameList.clear();
	m_takeNameList.clear();
	m_CommentList.clear();
}

void QmlMainWindow::disconnectProject()
{
	disconnectCameras();
}

void QmlMainWindow::setCurrentMode(int viewMode)
{
	m_viewMode = (ViewMode) viewMode;
}

int QmlMainWindow::getBlendMode()
{
	if (sharedImageBuffer->isWeightMapEditEnabled())
	{
		return WEIGHTVISUALIZER_MODE;
	}
	else
	{
		if (getGlobalAnimSetting().m_blendingMode == GlobalAnimSettings::Feathering)
			return FEATHER_MODE;
		else if (getGlobalAnimSetting().m_blendingMode == GlobalAnimSettings::MultiBandBlending)
			return MULTIBAND_MODE;
		else if (getGlobalAnimSetting().m_blendingMode == GlobalAnimSettings::WeightMapVisualization)
			return WEIGHTVISUALIZER_MODE;
		return MULTIBAND_MODE;
	}
}

void QmlMainWindow::setBlendMode(int blendMode)
{
	GlobalAnimSettings::BlendingMode internalBlendingMode;
	if (blendMode == FEATHER_MODE)
		internalBlendingMode = GlobalAnimSettings::Feathering;
	else if (blendMode == MULTIBAND_MODE)
		internalBlendingMode = GlobalAnimSettings::MultiBandBlending;
	else if (blendMode == WEIGHTVISUALIZER_MODE)
		internalBlendingMode = GlobalAnimSettings::WeightMapVisualization;

	if (!sharedImageBuffer->isWeightMapEditEnabled())
	{
		getGlobalAnimSetting().m_blendingMode = internalBlendingMode;
	}

	emit getGlobalAnimSetting().fireEventBlendSettingUpdated(internalBlendingMode, getGlobalAnimSetting().m_multiBandLevel);
}

void QmlMainWindow::setPlayMode(int playMode)
{
	m_playMode = (PlayMode)playMode;

	bool isPause = (m_playMode == START_MODE);

	if (!isPause && m_isPlayFinished)
	{
		sharedImageBuffer->getStitcher()->initForReplay();
		if (m_streamProcess)
		{
			if (m_streamProcess->isOpened())
				streamClose();			
			m_streamProcess->initializeStream(0);
		}

		// For replay function, Camera threads should be re-started...
		for (unsigned i = 0; i < cameraModuleMap.size(); ++i)
		{
			if (cameraModuleMap[i]->isConnected())
				cameraModuleMap[i]->startThreads(true);		// Must start after stream thread started!
		}

		// For Nodal
		for (unsigned i = 0; i < m_nodalCameraModuleMap.keys().size(); i ++)
		{
			if (m_nodalCameraModuleMap.values()[i]->isConnected())
				m_nodalCameraModuleMap.values()[i]->startThreads(true);
		}

		for (int i = 0; i < m_bannerThreads.size(); i++)
		{
			BannerThread* banner = m_bannerThreads[i];
			if (banner->isRunning())
			{
				banner->terminate();
				banner->wait();
			}				
			banner->reconnect();
			banner->start();
		}

		m_isPlayFinished = false;
	}

	// Play/Pause capture thread
	for (unsigned i = 0; i < cameraModuleMap.size(); ++i)
	{
		cameraModuleMap[i]->getCaptureThread()->playAndPause(isPause);
	}

	for (unsigned i = 0; i < m_nodalCameraModuleMap.keys().size(); i++)
	{
		m_nodalCameraModuleMap.values()[i]->getCaptureThread()->playAndPause(isPause);
	}

	// Play/Pause stitcher thread
	sharedImageBuffer->getStitcher()->playAndPause(isPause);
	
	// Play/Pause streamer thread
	if (sharedImageBuffer->getStreamer())
		sharedImageBuffer->getStreamer()->playAndPause(isPause);

	// Play/Pause banner thread
	for (int i = 0; i < m_bannerThreads.size(); i++)
		m_bannerThreads[i]->playAndPause(isPause);
}

void QmlMainWindow::openProject()
{
	if (m_reconnectCameras == false || m_isClose) {
		m_reconnectCameras = true;		
		connectToCameras();
		emit cameraResolutionChanged();
		m_isDisconnectComplete = false;
	}
}

bool QmlMainWindow::attach(CameraInput& camSettings, int deviceNumber, float startframe, float endframe, float fps)
{
	//
	// Add created ImageBuffer to SharedImageBuffer object
	//
	sharedImageBuffer->add(deviceNumber, true);

	GlobalState s;
	s.m_curFrame = startframe;

	sharedImageBuffer->addState(deviceNumber, s, true);

	//Create Ticket 
	// For Multi-Nodal,  used m_nodalDeviceNumberMap, be used for locking on Capture Thread
	sharedImageBuffer->addCamera(deviceNumber);

	//
	// Create View 
	//
	PANO_LOG("New Camera module");
	cameraModuleMap[deviceNumber] = new CameraModule(deviceNumber, sharedImageBuffer, this);
	PANO_LOG(QString("Connecting Cam (FPS: %1, WxH: %2x%3)").ARGN(camSettings.fps).ARGN(camSettings.xres).ARGN(camSettings.xres));
	if (cameraModuleMap[deviceNumber]->connectToCamera(camSettings.xres, camSettings.yres,
		sharedImageBuffer->getGlobalAnimSettings()->m_captureType))
	/*if (cameraModuleMap[deviceNumber]->connectToCamera(camSettings.xres, camSettings.yres,
		camSettings.cameraType))*/
	{
		PANO_LOG("Connected Camera");
		connect(cameraModuleMap[deviceNumber]->getCaptureThread(), SIGNAL(firstFrameCaptured(int)), this, SLOT(setPlayMode(int)));
	}
	else
	{
		PANO_ERROR(QString("[%1] Failed to connect camera!").ARGN(deviceNumber));
		sharedImageBuffer->setViewSync(deviceNumber, false);
		return false;
	}

	return true;
}

bool QmlMainWindow::attachNodalCamera(CameraInput& camSettings, int deviceNumber, int nForegroundCameras)
{
	int nodalDeviceNumber = NODAL_CAMERA_INDEX + deviceNumber;
	sharedImageBuffer->add(nodalDeviceNumber, true);

	GlobalState s;
	s.m_curFrame = 0;
	sharedImageBuffer->addState(nodalDeviceNumber, s, true);

	sharedImageBuffer->addCamera(nodalDeviceNumber);

	//
	// Create View 
	//
	PANO_LOG("New Camera module");
	m_nodalCameraModuleMap[nodalDeviceNumber] = new CameraModule(nodalDeviceNumber, sharedImageBuffer, this);
	PANO_LOG(QString("Connecting Camera"));
	if (m_nodalCameraModuleMap[nodalDeviceNumber]->connectToCamera(camSettings.xres, camSettings.yres, camSettings.cameraType))
	{
		PANO_LOG("Connected Camera");
		connect(m_nodalCameraModuleMap[nodalDeviceNumber]->getCaptureThread(), SIGNAL(firstFrameCaptured(int)), this, SLOT(setPlayMode(int)));
	}
	else
	{
		PANO_ERROR(QString("[N] Failed to connect camera!"));
		return false;
	}

	return true;
}

void QmlMainWindow::setPlaybackMode(int playMode)
{
	m_playbackMode = (PlayMode)playMode;

	if (m_playbackMode == PAUSE_MODE)
		m_isStopPlayback = false;

	bool isPause = (m_playbackMode == START_MODE);
	
	// Play/Pause capture thread
	if (m_playbackModule)
		m_playbackModule->getPlaybackThread()->playAndPause(isPause);

	// Play/Pause stitcher thread
	sharedImageBuffer->getPlaybackStitcher()->playAndPause(isPause);

	// Play/Pause streamer thread
	if (sharedImageBuffer->getStreamer())
		sharedImageBuffer->getStreamer()->playAndPause(isPause);
}

void QmlMainWindow::setSeekValue(int nValue)
{
	m_isStopPlayback = false;

	struct ThreadStatisticsData statsData;
	statsData.nFramesProcessed = nValue;
	setETimeValue(statsData.toString(m_d360Data.getFps(), 0));

	m_playbackModule->getPlaybackThread()->playAndPause(true);
	sharedImageBuffer->getPlaybackStitcher()->playAndPause(true);

	bool bPlayMode = (m_playbackMode == START_MODE);
	m_playbackModule->getPlaybackThread()->setSeekFrames(nValue, bPlayMode);
	sharedImageBuffer->getPlaybackStitcher()->setSeekFrames(nValue, bPlayMode);
	sharedImageBuffer->setSeekFrames(nValue);
	Sleep(100);
	m_playbackModule->getPlaybackThread()->playAndPause(false);
	sharedImageBuffer->getPlaybackStitcher()->playAndPause(false);
}

void QmlMainWindow::setCurIndex(QModelIndex curIndex)
{
	m_curIndex = curIndex;
	emit curIndexChanged(m_curIndex);
}

void QmlMainWindow::stopPlayback()
{
	setETimeValue("00:00:00.00");
	setSliderValue(0);

	m_isStopPlayback = true;

	if (m_playbackModule)
	{
		setPlaybackMode(START_MODE);
		m_playbackModule->startThreads(true);
	}
}

void QmlMainWindow::setPlaybackBackward()
{
	struct ThreadStatisticsData statsData = sharedImageBuffer->getPlaybackStitcher()->getStatisticsData();
	int nFramesProcessed = statsData.nFramesProcessed - 6;
	if (nFramesProcessed < 1) nFramesProcessed = 0;

	setPlaybackMode(START_MODE);
	setSliderValue(nFramesProcessed);
	setSeekValue(nFramesProcessed);
}

void QmlMainWindow::setPlaybackForward()
{
	struct ThreadStatisticsData statsData = sharedImageBuffer->getPlaybackStitcher()->getStatisticsData();
	int nFramesProcessed = statsData.nFramesProcessed + 4;
	if (nFramesProcessed > m_nDurationFrame) nFramesProcessed = m_nDurationFrame;

	setPlaybackMode(START_MODE);
	setSliderValue(nFramesProcessed);
	setSeekValue(nFramesProcessed);
}

void QmlMainWindow::setPlaybackPrev()
{
	QModelIndex prevIndex = m_qmlTakeManagement->prevModelIndex(m_curIndex);
	if (!prevIndex.isValid())
		return;

	setCurIndex(prevIndex);
	resetPlayback(prevIndex);

	if (m_playbackModule)
	{
		setETimeValue("00:00:00.00");
		setSliderValue(0);

		setPlaybackMode(START_MODE);
		m_playbackModule->startThreads(true);
	}
}

void QmlMainWindow::setPlaybackNext()
{
	QModelIndex nextIndex = m_qmlTakeManagement->nextModelIndex(m_curIndex);
	if (!nextIndex.isValid())
		return;

	setCurIndex(nextIndex);
	resetPlayback(nextIndex);

	if (m_playbackModule)
	{
		setETimeValue("00:00:00.00");
		setSliderValue(0);

		setPlaybackMode(START_MODE);
		m_playbackModule->startThreads(true);
	}
}

void QmlMainWindow::disconnectPlayback()
{
	if (m_playbackModule) {
		setPlaybackMode(START_MODE);
		if (!m_playbackModule->isConnected())
			m_playbackModule->getPlaybackThread()->forceFinish();
		
		delete m_playbackModule;
		m_playbackModule = NULL;
	}

	std::shared_ptr< PlaybackStitcher > stitcher = m_process->getPlaybackStitcherThread();
	if (!stitcher->isFinished())
		stitcher->stopStitcherThread();
}

bool QmlMainWindow::connectPlayback()
{
	sharedImageBuffer->add(PLAYBACK_CAMERA_INDEX, true);

	GlobalState s;
	s.m_curFrame = 0;
	sharedImageBuffer->addState(PLAYBACK_CAMERA_INDEX, s, true);

	std::shared_ptr< PlaybackStitcher > stitcher = m_process->getPlaybackStitcherThread();
	sharedImageBuffer->setPlaybackStitcher(stitcher);

	GlobalAnimSettings& gasettings = m_d360Data.getGlobalAnimSettings();
	//
	// Create View 
	//
	PANO_LOG("New Playback module");
	m_playbackModule = new PlaybackModule(sharedImageBuffer, this);
	PANO_LOG(QString("Connecting Playback"));
	if (m_playbackModule->connectToPlayback(gasettings.m_panoXRes, gasettings.m_panoYRes, D360::Capture::CaptureDomain::CAPTURE_VIDEO))
	{
		PANO_LOG("Connected Playback");
		connect(m_playbackModule->getPlaybackThread(), SIGNAL(firstFrameCaptured(int)), this, SLOT(setPlaybackMode(int)));

		if (m_playbackModule && m_playbackModule->isConnected())
			m_playbackModule->startThreads();
	}
	else
	{
		PANO_ERROR(QString("[N] Failed to connect Playback!"));
		return false;
	}

	PANO_LOG(">Initializing playback-stitcher ...");
	emit loadingMessageReceived("Initializing playback-stitching engine...");
	stitcher->init(QOpenGLContext::globalShareContext());
	PANO_LOG(">Starting playback-stitcher thread ...");
	stitcher->startThread();

	return true;
}

void QmlMainWindow::disconnectCameras()
{
	m_isAboutToStop = true;	

	PANO_LOG("Terminating audio and camera capture threads...");
	QMapIterator<int, AudioThread*> iter(audioModuleMap);
	while (iter.hasNext()) {
		iter.next();

		PANO_LOG(QString("[%1] Stop audio thread.").ARGN(iter.key()));
		emit loadingMessageReceived(QString("Stopping audio engine #%1...").ARGN(iter.key()));

		AudioThread* audio = iter.value();
		audio->stopAudioThread();
		delete audio;
	}
	audioModuleMap.clear();

	bool isAllCameraFinished = true; // if all cameras finished to play, all threads were all stopped. So that finishedThread() function is NOT accessed.
	for (QMap<int, CameraModule *>::const_iterator itr = cameraModuleMap.begin(); itr != cameraModuleMap.end(); ++itr)
	{		
		PANO_LOG(QString("[%1] Stop camera thread.").ARGN(itr.key()));
		emit loadingMessageReceived(QString("Disconnecting from camera #%1...").ARGN(itr.key()));

		CameraModule * camera = itr.value();
		if (!camera->isConnected()) {
			PANO_LOG(QString("[%1]  Force finish capture thread (was NOT connected)").ARGN(itr.key()));
			camera->getCaptureThread()->forceFinish();
		}
		isAllCameraFinished = isAllCameraFinished && camera->getCaptureThread()->isFinished();
		delete camera;
	}

	cameraModuleMap.clear();
	deviceNumberMap.clear();

	if (m_nodalCameraModuleMap.keys().size() != 0)
	{
		PANO_LOG(QString("[N] Stop camera thread."));
		emit loadingMessageReceived(QString("Disconnecting from camera..."));

		for (unsigned i = 0; i < m_nodalCameraModuleMap.keys().size(); i ++)
		{
			if (!m_nodalCameraModuleMap.values()[i]->isConnected()) {
				PANO_LOG(QString("[N]  Force finish capture thread (was NOT connected)"));
				m_nodalCameraModuleMap.values()[i]->getCaptureThread()->forceFinish();
			}
			isAllCameraFinished = isAllCameraFinished && m_nodalCameraModuleMap.values()[i]->getCaptureThread()->isFinished();

			// [only to delete on Map is Possible]
			delete m_nodalCameraModuleMap.values()[i];
		}		
	}

	PANO_LOG("Terminating banner threads...");
	// stop banner thread
	for (int i = 0; i < m_bannerThreads.size(); i++)
	{
		emit loadingMessageReceived(QString("Disconnecting from banner camera #%1...").ARGN(i));

		BannerThread* banner = m_bannerThreads[i];
		if (!banner->isConnected())
		{
			PANO_LOG(QString("Force finish banner thread (was NOT connected)"));
			banner->forceFinish();
		}
		banner->stopBannerThread();
		/*if (!banner->isFinished())
			banner->waitForFinish();*/
	}

	// removed m_istarted condition because cameras can't be connected because of some reasons.
	/*if (m_isEndCaptureThreads && m_deletedAudioCnt == 0 && m_isStarted) {*/
	//if (m_deletableAudioCnt == 0) {
	//m_isStarted = false;
	//emit started(false);			// Sending started signal to QML.		
	//Only must be stop stitcher thread after stop all thread associated with cameras.
	//}

	m_isPlayFinished = false;

	PANO_LOG("Stopping stitcher thread...");
	if (!m_process->getStitcherThread()->isFinished() && isAllCameraFinished)
		m_process->getStitcherThread()->stopStitcherThread();
}

void QmlMainWindow::disconnectCamera(int index)
{
	// Local variable(s)
	bool doDisconnect = true;

	//
	// Disconnect camera
	//
	if (doDisconnect)
	{
		if (audioModuleMap.contains(index)) {
			audioModuleMap[index]->stopAudioThread();
			audioModuleMap.remove(index);
		}
	}
}

void QmlMainWindow::updateCameraView(QObject* camView, int deviceNum)
{
	GlobalAnimSettings::CameraSettingsList& cameraSettings = m_d360Data.getGlobalAnimSettings().cameraSettingsList();
	if (deviceNum < 0 || deviceNum >= cameraSettings.size())
		return;
	
	MCQmlCameraView *cameraView = qobject_cast<MCQmlCameraView*>(camView);
	cameraView->setCameraNumber(deviceNum);
	cameraView->setSharedImageBuffer(sharedImageBuffer);
	//cameraView->init(true, NULL);
	cameraView->init(MCQmlCameraView::CameraViewType_Standard, NULL);

	QString title = "";
	QString path = "";
	QString camName = "";
	camName = cameraSettings[deviceNum].name;
	if (m_d360Data.getGlobalAnimSettings().m_captureType == D360::Capture::CAPTURE_DSHOW)
	{
		camName = getCameraDeviceNameByPath(camName);
	}
	else
	{
		for (int i = camName.length() - 1; i > 0; i--)
		{
			if (camName.at(i) == '/')
			{
				path = camName.mid(0, i);
				title = camName.mid(i + 1, camName.length() - path.length() - 5);
				break;
			}
		}
	}
	if (title == "")
	{
		cameraView->setCameraName(camName);
	}
	else
	{
		cameraView->setCameraName(title);
	}
}

void QmlMainWindow::disableCameraView(QObject* camView)
{
	MCQmlCameraView *cameraView = qobject_cast<MCQmlCameraView*>(camView);
	cameraView->setCameraNumber(-1);
	cameraView->setSharedImageBuffer(NULL);
	//cameraView->init(true, NULL);
	cameraView->init(MCQmlCameraView::CameraViewType_Standard, NULL);
	cameraView->setCameraName("");
}

void QmlMainWindow::enableCameraView(QObject* camView, bool enabled)
{
	MCQmlCameraView *cameraView = qobject_cast<MCQmlCameraView*>(camView);
	cameraView->enableView(enabled);
}

void QmlMainWindow::updateStitchView(QObject* stitchView)
{
	MCQmlCameraView *stitcherView = qobject_cast<MCQmlCameraView*>(stitchView);
	m_stitcherView = stitcherView;
	m_stitcherView->setSharedImageBuffer(sharedImageBuffer);
	m_stitcherView->init(MCQmlCameraView::CameraViewType_Stitch, NULL);
	stitcherView->setCameraName("StitchView");
}

void QmlMainWindow::updateCalibCameraView(QObject* calibView)
{
	MCQmlCameraView *calibrationView = qobject_cast<MCQmlCameraView*>(calibView);
	m_calibView = calibrationView;
	m_calibView->setSharedImageBuffer(sharedImageBuffer);
	m_calibView->init(MCQmlCameraView::CameraViewType_Calibrate, NULL);
	calibrationView->setCameraName("CalibView");
}

void QmlMainWindow::updatePlaybackView(QObject* stitchView)
{
	MCQmlCameraView *playbackView = qobject_cast<MCQmlCameraView*>(stitchView);
	m_playbackView = playbackView;
	m_playbackView->setSharedImageBuffer(sharedImageBuffer);
	m_playbackView->init(MCQmlCameraView::CameraViewType_Playback, NULL);
	m_playbackView->setCameraName("PlaybackView");
}

void QmlMainWindow::updateFullScreenStitchView(QObject* stitchView, int screenNo)
{
	MCQmlCameraView *stitcherView = qobject_cast<MCQmlCameraView*>(stitchView);	
	m_stitcherFullScreenView = stitcherView;
	m_stitcherFullScreenView->setSharedImageBuffer(sharedImageBuffer);

	QRect rect = getFullScreenInfo(screenNo);
	//m_stitcherFullScreenView->init(false, NULL);	
	m_stitcherFullScreenView->init(MCQmlCameraView::CameraViewType_Stitch, NULL);
	m_stitcherFullScreenView->setCameraName("FullScreen");
	setFullScreenStitchViewMode(screenNo);
}

void QmlMainWindow::setFullScreenStitchViewMode(int screenNo)
{
	if (m_stitcherFullScreenView)
	{
		QRect rect = getFullScreenInfo(screenNo);
		m_stitcherFullScreenView->setFullScreenMode(true, rect.width(), rect.height());
	}		
}

void QmlMainWindow::updateInteractView(QObject* interactView)
{
	QmlInteractiveView *interactiveView = qobject_cast<QmlInteractiveView*>(interactView);
	m_interactView = interactiveView;
	m_interactView->setSharedImageBuffer(sharedImageBuffer);
}

void QmlMainWindow::updatePreView(QObject* preView)
{
	MCQmlCameraView *preViewer = qobject_cast<MCQmlCameraView*>(preView);
	preViewer->setSharedImageBuffer(sharedImageBuffer);
	//preViewer->init(false, NULL);
	preViewer->init(MCQmlCameraView::CameraViewType_Stitch, NULL);
	preViewer->setCameraName("PreView");
}

int QmlMainWindow::openIniPath(QString iniPath)
{
	QString path;
	int index = -1;
	PANO_LOG("Open Ini File " + iniPath);
	if (!openIniFileAndUpdateUI(iniPath))
	{
		PANO_N_ERROR("This version of configuration is not supported.");
		return index;
	}

	if (iniPath.mid(iniPath.lastIndexOf(".") + 1) == QString("ini"))
		return index;

	QString m_recentTitle = iniPath.mid(iniPath.lastIndexOf("/") + 1,
		iniPath.lastIndexOf(".") - iniPath.lastIndexOf("/") - 1);
	m_recent = { m_recentTitle, iniPath, m_d360Data.getCaptureType() };

	if (m_recentList.contains(m_recent)) {
		index = m_recentList.indexOf(m_recent);
		m_recentList.removeAt(index);
	}
	else {
		index = 0;
	}
	m_recentList.insert(0, m_recent);

	return index;
}

void QmlMainWindow::saveWeigthMap(QString iniPath)
{
	std::shared_ptr<D360Stitcher> stitcher = sharedImageBuffer->getStitcher();
	stitcher->lockBannerMutex();
	stitcher->saveBanner(m_d360Data.getGlobalAnimSettings());
	stitcher->unlockBannerMutex();
	if (iniPath == "")
		iniPath = m_recent.fullPath;

	if (m_isStarted)
		stitcher->saveWeightMaps(iniPath);
}

int QmlMainWindow::saveIniPath(QString iniPath)
{
	PANO_N_LOG("Saving configuration...");
	ConfigZip confZip;
	QString path;
	int index = -1;
	if (iniPath == "")
	{
		if (QFile::exists(m_recent.fullPath))
			QFile::remove(m_recent.fullPath);

		path = m_recent.fullPath.left(m_recent.fullPath.lastIndexOf("/")) + "/configuration.ini";

		PANO_LOG("Save l3d File " + m_recent.fullPath);
		
		m_d360Data.saveINI(m_recent.fullPath, path);
		confZip.createZipFile(m_recent.fullPath, path);
		index = m_recentList.contains(m_recent);
	}
	else
	{
		if (QFile::exists(iniPath))
			QFile::remove(iniPath);

		path = iniPath.left(iniPath.lastIndexOf("/")) + "/configuration.ini";
		PANO_LOG("Save l3d File " + iniPath);

		m_d360Data.saveINI(iniPath,path);
		confZip.createZipFile(iniPath, path);

		QString m_recentTitle = iniPath.mid(iniPath.lastIndexOf("/") + 1, iniPath.lastIndexOf(".") - iniPath.lastIndexOf("/") - 1);
		m_recent = { m_recentTitle, iniPath, m_d360Data.getCaptureType() };

		if (m_recentList.contains(m_recent))
		{
			index = m_recentList.indexOf(m_recent);
			m_recentList.removeAt(index);
		}
		else
		{
			index = 0;
		}
		m_recentList.insert(0, m_recent);
	}

	saveWeigthMap(iniPath);

	PANO_N_LOG("Saved successfully!");

	return index;
}

QString QmlMainWindow::getRecentPath(int index)
{
	QString path;
	if (m_recentList[index].fullPath.length() > 30)
	{
		path = m_recentList[index].fullPath.left(27) + "...";
	}
	else{
		path = m_recentList[index].fullPath;
	}
	return path;
}

QString QmlMainWindow::getDisplayName(QString l3dPath)
{
	QString displayNameStr = "";
	ConfigZip cfz;
	displayNameStr = cfz.getValue(l3dPath,"D360","name");
	if (displayNameStr.isEmpty())
		displayNameStr = l3dPath.mid(l3dPath.lastIndexOf("/") + 1);
	displayNameStr = displayNameStr.left(displayNameStr.lastIndexOf("."));
	return displayNameStr;
}

// type:0(video), type:1(audio)
int QmlMainWindow::getDeviceIndex(QString name, int type)
{
	int index = -1;
	if (type == 0)
	{
        QMapIterator<int, DeviceMetaInformation> iter(m_videoDevices);
		while (iter.hasNext()) {
			iter.next();
            //if (iter.value().path.trimmed() == name.trimmed()) {
			if(iter.value().path == name.toStdString()){
				index = iter.key();
				break;
			}
		}
	}
	else if (type == 1)
	{
		QMapIterator<int, QString> iter(m_audioDevices);
		while (iter.hasNext()) {
			iter.next();
			if (iter.value().trimmed() == name.trimmed()) {
				index = iter.key();
				break;
			}
		}
	}

	return index;
}

QString QmlMainWindow::getMicDeviceName(int index)
{
	QString name = "";
	if (m_audioDevices.contains(index))
		name = m_audioDevices[index];

	return name;
}

QString QmlMainWindow::getCameraDeviceName(int index)
{
	QString name = "";
    if (m_videoDevices.contains(index))
        name = QString::fromStdString(m_videoDevices[index].name);

	return name;
}

QString QmlMainWindow::getCameraDevicePath(int index)
{
	QString path = "";
    if (m_videoDevices.contains(index))
        path = QString::fromStdString( m_videoDevices[index].path );

	return path;
}

QString QmlMainWindow::getCameraDeviceNameByPath(QString devicePath)
{
	int dupIndex = 0;
	return getDeviceNameByDevicePath(devicePath, dupIndex);
}

QString QmlMainWindow::getDeviceNameByDevicePath(QString devicePath, int& dupIndex)
{
	QString deviceName = "";
	int deviceIndex = -1;
    for (int i = 0; i < m_videoDevices.size(); i++)
	{
        if (m_videoDevices[i].path == devicePath.toStdString())
		{
			deviceName = QString::fromStdString(m_videoDevices[i].name);
			deviceIndex = i;
			break;
		}
	}

	dupIndex = 0;
	for (int i = 0; i < deviceIndex; i++)
	{
        if (m_videoDevices[i].name == deviceName.toStdString())
			dupIndex++;
	}

	return deviceName;
}

QObject* QmlMainWindow::recDialog() const
{
	return m_qmlRecentDialog;
}

void QmlMainWindow::setRecDialog(QObject *recDialog)
{
	if (m_qmlRecentDialog == recDialog)
		return;

	QmlRecentDialog *s = qobject_cast<QmlRecentDialog*>(recDialog);
	if (!s)
	{
		PANO_LOG("Source must be a QmlRecentDialog type");
		return;
	}
	m_qmlRecentDialog = s;

	emit recDialogChanged(m_qmlRecentDialog);
}

void QmlMainWindow::saveRecentMgrToINI()
{
	QString recentfilepath = QDir::currentPath() + QLatin1String("/RecentManagement.ini");
	
	if (QFile(recentfilepath).exists())
		QFile(recentfilepath).remove();
	QSettings settings(recentfilepath, QSettings::IniFormat);
	settings.beginGroup("Information");
	settings.setValue(QLatin1String("count"), (m_recentList.size()));
	settings.endGroup();
	for (int i = 0; i < m_recentList.size(); i++)
	{
		QString recentName = QString("PanoOne_%1").ARGN(i);
		settings.beginGroup(recentName);
		settings.setValue(QLatin1String("Title"), (m_recentList[i].title));
		settings.setValue(QLatin1String("Path"), (m_recentList[i].fullPath));
		settings.setValue(QLatin1String("CaptureType"), (m_recentList[i].type));
		settings.endGroup();
	}
	settings.sync();
}

bool QmlMainWindow::openRecentMgrToINI()
{
	QString recentfilepath = QDir::currentPath() + QLatin1String("/RecentManagement.ini");
	if (!QFile(recentfilepath).exists())
	{
		PANO_N_ERROR("RecentManagement.ini doesn't exist!");
		return false;
	}
	int count = 0;
	QSettings settings(recentfilepath, QSettings::IniFormat);
	{
		settings.beginGroup("Information");
		const QStringList childKeys = settings.childKeys();
		foreach(const QString &childKey, childKeys)
		{
			if (childKey == "count")
				count = settings.value(childKey).toInt();
		}
		settings.endGroup();
		for (int i = 0; i < count; i++)
		{
			QString recentName = QString("PanoOne_%1").ARGN(i);

			settings.beginGroup(recentName);
			const QStringList childKeys = settings.childKeys();
			RecentInfo info;
			foreach(const QString &childKey, childKeys)
			{
				if (childKey == "Title")
					info.title = settings.value(childKey).toString();
				if (childKey == "Path")
					info.fullPath = settings.value(childKey).toString();
				if (childKey == "CaptureType")
					info.type = settings.value(childKey).toInt();
			}
			m_recentList.append(info);
			settings.endGroup();
		}
	}
	return true;
}

bool QmlMainWindow::openIniFileAndUpdateUI(QString projFile)
{
	// open ini file
	if (!m_d360Data.parseINI(projFile))
		return false;

	// update UI
	emit grayLutDataChanged(m_d360Data.getGlobalAnimSettings().getLutData()[0]);
	emit redLutDataChanged(m_d360Data.getGlobalAnimSettings().getLutData()[1]);
	emit greenLutDataChanged(m_d360Data.getGlobalAnimSettings().getLutData()[2]);
	emit blueLutDataChanged(m_d360Data.getGlobalAnimSettings().getLutData()[3]);

	// initialize for Live
	if (m_d360Data.getGlobalAnimSettings().m_haveLiveBackground)
		initTemplateCameraObject();
	
	// [foreground for Live][without Video Background, i.e. background is Live, foreground is also Live]
	int numForegroundCameras = 0;
	if (m_d360Data.getGlobalAnimSettings().m_haveLiveBackground)
	{
		for (unsigned i = 0; i < m_d360Data.getGlobalAnimSettings().cameraSettingsList().size(); i++)
		{
			int foregroundSlotIndex = m_d360Data.getGlobalAnimSettings().m_foregroundSlotIndexMap.values()[i];
			sendCameraName(foregroundSlotIndex, m_d360Data.getGlobalAnimSettings().cameraSettingsList()[i].name, false, false);
			setTempStereoType(foregroundSlotIndex, m_d360Data.getGlobalAnimSettings().cameraSettingsList()[i].stereoType);
			setForegroundSlot(foregroundSlotIndex);
		}
	}

	// [foreground]
	if (!m_d360Data.getGlobalAnimSettings().m_haveLiveBackground)
	{
		for (unsigned i = 0; i < m_d360Data.getCameraCount(); i++)
		{
			if (m_d360Data.getGlobalAnimSettings().m_foregroundSlotIndexMap.size() < 1)
				continue;
			int foregroundSlotIndex = m_d360Data.getGlobalAnimSettings().m_foregroundSlotIndexMap.values()[i];
			//sendCameraName(foregroundSlotIndex, m_d360Data.getGlobalAnimSettings().cameraSettingsList()[i].name, false, false);
			sendVideoPath(foregroundSlotIndex, m_d360Data.getGlobalAnimSettings().cameraSettingsList()[i].name);
			setTempStereoType(foregroundSlotIndex, m_d360Data.getGlobalAnimSettings().cameraSettingsList()[i].stereoType);
			setTempAudioSettings(foregroundSlotIndex, m_d360Data.getGlobalAnimSettings().cameraSettingsList()[i].audioType);

			setForegroundSlot(foregroundSlotIndex);
		}
	}	

	// [background]
	for (int i = 0; i < m_d360Data.getGlobalAnimSettings().m_nodalVideoFilePathMap.values().size(); i++)
	{
		int backgroundSlotIndex = m_d360Data.getGlobalAnimSettings().m_nodalSlotIndexMap.values()[i];
		setNodalVideoFilePath(backgroundSlotIndex, m_d360Data.getGlobalAnimSettings().m_nodalVideoFilePathMap[backgroundSlotIndex]);
		setNodalMaskImageFilePath(backgroundSlotIndex, m_d360Data.getGlobalAnimSettings().m_nodalMaskImageFilePathMap[backgroundSlotIndex]);

		if (m_d360Data.getGlobalAnimSettings().m_haveLiveBackground) {
			setNodalCameraIndex(backgroundSlotIndex);
		}
		else {
			setNodalCameraIndex(-1);
		}
		setNodalSlotIndex(backgroundSlotIndex);
	}

	if (m_d360Data.getGlobalAnimSettings().m_haveLiveBackground)
	{
		QList<QString> strList; strList.clear();
		QList<int> numList; numList.clear();

		for (int i = 0; i < m_cameraNameList.values().size(); i ++)
		{
			strList.append(m_cameraNameList.values()[i]);
			//numList.append(i);
		}

		openTemplateCameraIniFile(strList, numList);
	}

	return true;
}

int QmlMainWindow::recentOpenIniPath(QString sourcepath)
{
	QString path;
	int index = -1;
	if (!QFile(sourcepath).exists()) {
		PANO_N_ERROR(QString("Ini file does not exist(%1).").arg(sourcepath));
		return FILE_NO_EXIST;
	}

	PANO_LOG("Start reading configuration file");
	if (!openIniFileAndUpdateUI(sourcepath))
	{
		PANO_N_ERROR("This version of configuration is not supported.");
		return false;
	}
	

	QString m_recentTitle = sourcepath.mid(sourcepath.lastIndexOf("/") + 1,
		sourcepath.lastIndexOf(".") - sourcepath.lastIndexOf("/") - 1);
	m_recent = { m_recentTitle, sourcepath, m_d360Data.getCaptureType() };

	if (m_recentList.contains(m_recent)) {
		index = m_recentList.indexOf(m_recent);
		m_recentList.removeAt(index);
	}
	else {
		index = 0;
	}
	m_recentList.insert(0, m_recent);

	PANO_LOG("Finished reading configuration file");

	return index;
}

void QmlMainWindow::clearTemplateSetting() 
{
	m_d360Data.getGlobalAnimSettings().setDefault();
	m_d360Data.getTempGlobalAnimSettings().setDefault();
	m_d360Data.clearTempStereoType();
}

int QmlMainWindow::openTemplateIniFile(QString iniFile)
{		
	m_templateIniFile = "";

	int ret = -1;
	if (iniFile == NULL || iniFile.isEmpty())
	{		
		return ret;
	}		

	QString path;
	if (!QFile(iniFile).exists()){
		PANO_N_ERROR(QString("Template Ini file does not exist(%1).").arg(iniFile));
		return FILE_NO_EXIST;
	}

	PANO_LOG("Start reading configuration file");

	if (!m_d360Data.parseINI(iniFile, true))
	{
		PANO_N_ERROR("This version of configuration is not supported.");
		return FILE_VERSION_NO_MATCH;
	}
	
	ret = 0;
	m_templateIniFile = iniFile;

	PANO_LOG("Finished reading configuration file");
	
	return ret;
}

void QmlMainWindow::setTemplateOrder(QList<QString> strSlotList, QList<int> orderList)
{
	openTemplateIniFile(m_templateIniFile);

	if (strSlotList.isEmpty() || orderList.isEmpty())
		return;	

	if (strSlotList.size() != orderList.size())
		return;

	m_templateOrderList.clear();
	for (int i = 0; i < orderList.size(); i++)
	{
		QString slot = strSlotList.at(i);
		int param = orderList.at(i);

		QMap<QString, int> slotOrderMap;
		slotOrderMap[slot] = param;
		m_templateOrderList.append(slotOrderMap);
	}
	
	m_d360Data.getTempGlobalAnimSettings().cameraSettingsList() = m_d360Data.getRigTemplateGlobalAnimSettings().cameraSettingsList();

	int temp = m_d360Data.getTempGlobalAnimSettings().cameraSettingsList().size();

	QMap<int, int> orgTempStereoTypeList = m_d360Data.getTempStereoTypeList();
	QList<int> newStereoTypeList;
	for (int i = 0; i < m_templateOrderList.size(); i++)
	{
		int order = (m_templateOrderList.at(i)).value(strSlotList.at(i));
		
		int newStereoType = 0;
		if (!orgTempStereoTypeList.contains(order))
			newStereoType = 0;
		else
			newStereoType = orgTempStereoTypeList[order];

		m_d360Data.setTempStereoType(i, newStereoType);

		m_d360Data.getTempGlobalAnimSettings().cameraSettingsList()[i] = m_d360Data.getRigTemplateGlobalAnimSettings().cameraSettingsList()[order];
	}
	
	m_d360Data.resetCameraSettings();
}

QList<QString> QmlMainWindow::getTemplateOrderMapList()
{
	QList<QString> result;
	for (int i = 0; i < m_templateOrderList.size(); i++)
	{
		QMap<QString, int> slotOrderMap = m_templateOrderList.at(i);
		QMapIterator<QString, int> mapIter(slotOrderMap);
		while (mapIter.hasNext()) {
			mapIter.next();
			result.append(QString("%1,%2").arg(mapIter.key()).ARGN(mapIter.value()));
		}
	}
	return result;
}

int QmlMainWindow::checkAudioExist(QString videoFilePath)
{
	BaseFfmpeg* baseFfmpeg = new BaseFfmpeg();
	AUDIO_CHANNEL_TYPE channelType = baseFfmpeg->checkAudioChannelInVideoFile(videoFilePath);
	delete baseFfmpeg;
	return channelType;
}

void QmlMainWindow::streamPanorama(unsigned char* panorama)
{
	// if (!m_oculusDevice) // commented by B
	if (/*m_offlineVideoSaveProcess && */m_streamProcess)
	{
		/*m_offlineVideoSaveProcess->streamPanorama(panorama);*/
		m_streamProcess->streamPanorama(panorama);
	}
}

void QmlMainWindow::streamAudio(int devNum, void* audioFrame)
{
	// if (!m_oculusDevice) // commented by B
	if (/*m_offlineVideoSaveProcess && */m_streamProcess)
	{
		/*m_offlineVideoSaveProcess->streamAudio(devNum, audioFrame);*/
		m_streamProcess->streamAudio(devNum, audioFrame);
	}
}

void QmlMainWindow::streamClose()
{	
	if (m_streamProcess && m_streamProcess->isOpened())
	{		
		m_streamProcess->disconnect();
		PANO_LOG("Stream closed");
	}
}

void QmlMainWindow::openTemplateCameraIniFile(QList<QString> strSlotList, QList<int> orderList)
{
	setTemplateOrder(strSlotList, orderList);

	m_d360Data.getGlobalAnimSettings().m_cameraCount = m_cameraNameList.size();

	m_d360Data.getTempGlobalAnimSettings().cameraSettingsList().resize(m_cameraNameList.size());
	for (int i = 0; i < m_cameraNameList.size(); i++)
	{
		m_d360Data.getTempGlobalAnimSettings().cameraSettingsList()[i].cameraType = (D360::Capture::CaptureDomain)m_videoDevices[i].type;
		m_d360Data.getTempGlobalAnimSettings().cameraSettingsList()[i].pixel_format = AV_PIX_FMT_BAYER_RGGB8;
	}
	if (!m_templateIniFile.isEmpty())
	{
		for (int i = 0; i < m_cameraNameList.size(); i++)
		{
			m_d360Data.getTempGlobalAnimSettings().cameraSettingsList()[i].name = m_cameraNameList[i];
		}
	}

	m_d360Data.initTemplateCamera(m_cameraNameList, m_audioNameList);
}

void QmlMainWindow::openTemplateVideoIniFile(QList<QString> strSlotList, QList<int> orderList)
{
	setTemplateOrder(strSlotList, orderList);

	m_d360Data.getGlobalAnimSettings().m_cameraCount = m_videoPathList.size();
	
	m_d360Data.getTempGlobalAnimSettings().cameraSettingsList().resize(m_videoPathList.size());
	if (!m_templateIniFile.isEmpty())
	{
		for (int i = 0; i < m_videoPathList.size(); i++)
		{
			m_d360Data.getTempGlobalAnimSettings().cameraSettingsList()[i].name = m_videoPathList[i];
		}
	}

	m_d360Data.initTemplateVideo(m_videoPathList);
}

void QmlMainWindow::openTemplateImageIniFile(QList<QString> strSlotList, QList<int> orderList)
{
	setTemplateOrder(strSlotList, orderList);

	m_d360Data.getGlobalAnimSettings().m_cameraCount = m_imagePathList.size();

	m_d360Data.getTempGlobalAnimSettings().cameraSettingsList().resize(m_imagePathList.size());
	if (!m_templateIniFile.isEmpty())
	{
		for (int i = 0; i < m_imagePathList.size(); i++)
		{
			m_d360Data.getTempGlobalAnimSettings().cameraSettingsList()[i].name = m_imagePathList[i];
		}
	}
	m_d360Data.initTemplateImage(m_imagePathList);
}

void QmlMainWindow::sendCameraName(int slotIndex, QString cameraName, bool isNodalInput, bool isVideoFile)
{
	m_cameraNameList[slotIndex] = cameraName;
	if (isNodalInput) {
		m_d360Data.getGlobalAnimSettings().m_nodalVideoIndex = slotIndex;
		m_d360Data.getGlobalAnimSettings().m_hasNodalOfflineVideo = isVideoFile;
	}
}

void QmlMainWindow::setLiveCamera(int slotIndex, QString strDevicePath)
{
	m_d360Data.getGlobalAnimSettings().m_totalCameraDevicesMap[slotIndex] = strDevicePath;
}

QString QmlMainWindow::getLiveCamera(int slotIndex)
{
	QString strCameraDevicePath = "";
	if (m_d360Data.getGlobalAnimSettings().m_totalCameraDevicesMap.contains(slotIndex))
		strCameraDevicePath = m_d360Data.getGlobalAnimSettings().m_totalCameraDevicesMap[slotIndex];

	return strCameraDevicePath;
}

void QmlMainWindow::setTotalIndexAndSelectedIndex(int totalIndex, int selectedIndex)
{
	m_d360Data.getGlobalAnimSettings().m_totalIndexAndSelectedIndexMap[totalIndex] = selectedIndex;
}

int QmlMainWindow::getSelectedIndex(int totalIndex) {
	int selectedIndex;
	if (m_d360Data.getGlobalAnimSettings().m_totalIndexAndSelectedIndexMap.contains(totalIndex))
		selectedIndex = m_d360Data.getGlobalAnimSettings().m_totalIndexAndSelectedIndexMap[totalIndex];

	return selectedIndex;
}

int QmlMainWindow::getLiveCameraCount()
{
	return m_d360Data.getGlobalAnimSettings().m_totalCameraDevicesMap.size();
}

void QmlMainWindow::setLiveStereo(int slotIndex, int stereoType)
{
	m_d360Data.getGlobalAnimSettings().m_totalStereoMap[slotIndex] = stereoType;
}

int QmlMainWindow::getLiveStereo(int slotIndex)
{
	int tempStereo;
	if (m_d360Data.getGlobalAnimSettings().m_totalStereoMap.contains(slotIndex))
		tempStereo = m_d360Data.getGlobalAnimSettings().m_totalStereoMap[slotIndex];

	return tempStereo;
}

void QmlMainWindow::sendAudioName(int slotIndex, QString audioName)
{
	m_audioNameList[slotIndex] = audioName;
}

void QmlMainWindow::sendVideoPath(int slotIndex, QString videoPath)
{	
	m_d360Data.getGlobalAnimSettings().m_videoFilePathMap[slotIndex] = videoPath;
	m_videoPathList[slotIndex] = videoPath;
}

void QmlMainWindow::setForegroundSlot(int slotIndex)
{
	m_d360Data.getGlobalAnimSettings().m_foregroundSlotIndexMap[slotIndex] = slotIndex;
}

void QmlMainWindow::sendImagePath(int slotIndex, QString imagePath)
{
	m_imagePathList[slotIndex] = imagePath;
}

void QmlMainWindow::Stop()
{

}

void QmlMainWindow::Pause()
{
	PANO_LOG("Terminating threads...");
	QMapIterator<int, AudioThread*> iter(audioModuleMap);
	while (iter.hasNext()) {
		iter.next();
		PANO_LOG(QString("[%1] Stop audio thread.").ARGN(iter.key()));
		AudioThread* audio = iter.value();
		audio->pause();
	}
	audioModuleMap.clear();
}

void QmlMainWindow::Start()
{

}

void QmlMainWindow::deleteRecentList(QString title)
{
	for (int i = 0; i < m_recentList.size(); i++)
	{
		if (m_recentList[i].title == title)
		{
			m_recentList.removeAt(i);
			break;
		}
	}
}

void QmlMainWindow::updateStitchingThreadStats(struct ThreadStatisticsData statData)
{
	QString strFps = QString::number(statData.averageFPS, 'f', 1) + " fps";
	if (m_streamProcess && m_streamProcess->isTakeRecording())
		setETimeValue(statData.toString(m_d360Data.getFps(), m_lastFramesProcessed));	
	else
		setETimeValue("00:00:00.00");
	setFpsValue(strFps);
}

void QmlMainWindow::updatePlaybackStitchingThreadStats(struct ThreadStatisticsData statData)
{
	QString strFps = QString::number(statData.averageFPS, 'f', 1) + " fps";
	if (statData.nFramesProcessed > 0 && !m_isStopPlayback) {
		setETimeValue(statData.toString(m_d360Data.getFps(), 0));
		setSliderValue(statData.nFramesProcessed - 1);
	} else {
		setETimeValue("00:00:00.00");
		setSliderValue(0);
	}
	setFpsValue(strFps);
}

void QmlMainWindow::setSliderValue(int value)
{
	m_nSliderValue = value;
	emit sliderValueChanged(m_nSliderValue);
}

void QmlMainWindow::setETimeValue(QString elapsedTime)
{
	m_eTimeValue = elapsedTime;
	emit elapsedTimeChanged(m_eTimeValue);
}

void QmlMainWindow::setFpsValue(QString fps)
{
	m_fpsValue = fps;
	emit fpsChanged(m_fpsValue);
}

void QmlMainWindow::enableOculus(bool oculusRift)
{
	if (m_d360Data.getGlobalAnimSettings().m_oculus == oculusRift)
		return;

	assert(m_interactView != NULL);
	if (m_interactView == NULL)
	{
		return;
	}

	int panoWidth = m_d360Data.getGlobalAnimSettings().m_panoXRes;
	int panoHeight = m_d360Data.getGlobalAnimSettings().m_panoYRes;
	std::shared_ptr< D360Stitcher > stitcher = m_process->getStitcherThread();

	//Added By I
	m_interactView->setOculusStatus(oculusRift);

	if (oculusRift) {
		if (m_oculusDevice == NULL) {
			m_oculusDevice = new OculusRender(stitcher->getContext());
			if (!m_oculusDevice->isCreated()) {
				delete m_oculusDevice;
				m_oculusDevice = NULL;
				PANO_N_ERROR("OculusRift device is not created!");
				return;
			}
			connect(m_oculusDevice, SIGNAL(finished(int, QString, int)), this, SLOT(finishedThread(int, QString, int))/*, Qt::DirectConnection*/);
			connect(m_oculusDevice, SIGNAL(started(int, QString, int)), this, SLOT(startedThread(int, QString, int)));
			//m_interactView->setOculusObject(m_oculusDevice);
			m_oculusDevice->initialize(sharedImageBuffer, panoWidth, panoHeight);
			m_oculusDevice->setPanoramaTexture(stitcher->getPanoramaTextureId());
			m_oculusDevice->installEventFilter(this);
		}
		else {
			if (!m_oculusDevice->isCreated()) {
				PANO_N_ERROR("OculusRift device is not created!");
				return;
			}
		}
		m_interactView->setOculusObject(m_oculusDevice);
		m_oculusDevice->startThread();
	}
	else {
		sharedImageBuffer->lockOculus();
		m_interactView->setOculusObject(NULL);
		sharedImageBuffer->unlockOculus();
		m_oculusDevice->stop(OculusRender::StopReason::SWITCH_OFF);
	}

	m_d360Data.getGlobalAnimSettings().m_oculus = oculusRift;
}

void QmlMainWindow::showChessboard(bool isShow)
{
	if (m_oculusDevice)
	{
		m_oculusDevice->showChessboard(isShow);
	}
}

void QmlMainWindow::setBlendLevel(int iBlendLevel)
{
	if (getGlobalAnimSetting().m_blendingMode == GlobalAnimSettings::MultiBandBlending)
	{
		getGlobalAnimSetting().m_multiBandLevel = iBlendLevel;
		emit getGlobalAnimSetting().fireEventBlendSettingUpdated(getGlobalAnimSetting().m_blendingMode, iBlendLevel);
	}
}

int QmlMainWindow::getBlendLevel()
{
	if (getGlobalAnimSetting().m_blendingMode == GlobalAnimSettings::MultiBandBlending)
	{
		return getGlobalAnimSetting().m_multiBandLevel;		
	}
	return 0;
}

float QmlMainWindow::getLeft(int iDeviceNum)
{
	return m_d360Data.getLeft(iDeviceNum);
}

void QmlMainWindow::setLeft(float fLeft, int iDeviceNum)
{
	m_d360Data.setLeft(fLeft, iDeviceNum);
}

float QmlMainWindow::getRight(int iDeviceNum)
{
	return m_d360Data.getRight(iDeviceNum);
}

void QmlMainWindow::setRight(float fRight, int iDeviceNum)
{
	m_d360Data.setRight(fRight, iDeviceNum);
}

float QmlMainWindow::getTop(int iDeviceNum)
{
	return m_d360Data.getTop(iDeviceNum);
}
void QmlMainWindow::setTop(float fTop, int iDeviceNum)
{
	m_d360Data.setTop(fTop, iDeviceNum);
}

float QmlMainWindow::getBottom(int iDeviceNum)
{
	return m_d360Data.getBottom(iDeviceNum);
}
void QmlMainWindow::setBottom(float fBottom, int iDeviceNum)
{
	m_d360Data.setBottom(fBottom, iDeviceNum);
}

int QmlMainWindow::getStereoType(int iIndex)
{
	return m_d360Data.getStereoType(iIndex);
}
void QmlMainWindow::setStereoType(int iStereoType, int iDeviceNum)
{
	m_d360Data.setStereoType(iStereoType, iDeviceNum);
}

void QmlMainWindow::setAudioType(int iAudioType, int iDeviceNum)
{
	m_d360Data.setAudioType(iAudioType, iDeviceNum);
}

void QmlMainWindow::onCalculatorGain()
{
	m_d360Data.getGlobalAnimSettings().refreshTempCameraSettingsList();
	sharedImageBuffer->getStitcher()->calcGain();
}

void QmlMainWindow::onRollbackGain()
{
	m_d360Data.getGlobalAnimSettings().rollbackCameraSettingsList();
	sharedImageBuffer->getStitcher()->restitch();
}

void QmlMainWindow::onResetGain()
{
	sharedImageBuffer->getStitcher()->resetGain();
}

void QmlMainWindow::setTempStereoType(int iIndex, int iStereoType)
{
	m_d360Data.setTempStereoType(iIndex, iStereoType);
}

int QmlMainWindow::getTempStereoType(int iIndex)
{
	return m_d360Data.getTempStereoType(iIndex);
}

void QmlMainWindow::setTempImagePath(int iIndex, QString fileDir)
{
	m_d360Data.setTempImagePath(iIndex, fileDir);
}

QString QmlMainWindow::getTempImagePath(int iIndex)
{
	return m_d360Data.getTempImagePath(iIndex);
}

void QmlMainWindow::setTempImagePrefix(int iIndex, QString filePrefix)
{
	m_d360Data.setTempImagePrefix(iIndex, filePrefix);
}

QString QmlMainWindow::getTempImagePrefix(int iIndex)
{
	return m_d360Data.getTempImagePrefix(iIndex);
}

void QmlMainWindow::setTempImageExt(int iIndex, QString fileExt)
{
	m_d360Data.setTempImageExt(iIndex, fileExt);
}

QString QmlMainWindow::getTempImageExt(int iIndex)
{
	return m_d360Data.getTempImageExt(iIndex);
}

void QmlMainWindow::setTempAudioSettings(int iIndex, int iAudioType)
{
	m_d360Data.setTempAudioSettings(iIndex, iAudioType);
}

int QmlMainWindow::getTempAudioSettings(int iIndex)
{
	return m_d360Data.getTempAudioSettings(iIndex);

}

int QmlMainWindow::getTempAudioSettingsEx(QString devName)
{
	for (unsigned i = 0; i < m_d360Data.getGlobalAnimSettings().cameraSettingsList().size(); i++)
	{
		if (m_d360Data.getGlobalAnimSettings().getCameraInput(i).audioName == devName)
			return m_d360Data.getGlobalAnimSettings().getCameraInput(i).audioType;
	}
	return 1;	// Default is Left mode.
}

void QmlMainWindow::onPressedSpherical(int iPosX, int iPosY)
{
	QPoint pos(iPosX, iPosY);
	if (m_stitcherView)
		m_stitcherView->onMousePress(pos);
}

void QmlMainWindow::onMovedSpherical(int iPosX, int iPosY)
{
	QPoint pos(iPosX, iPosY);
	if (m_stitcherView)
		m_stitcherView->onMouseMove(pos);
}

void QmlMainWindow::onReleasedSpherical(int iPosX, int iPosY)
{
	QPoint pos(iPosX, iPosY);
	if (m_stitcherView)
		m_stitcherView->onMouseRelease(pos);
}

void QmlMainWindow::onDoubleClickedSpherical(int iPosX, int iPosY)
{
	GlobalAnimSettings& setting = g_mainWindow->getGlobalAnimSetting();

	double panoW = getPanoXres();
	double panoH = getPanoYres();
	double winW = m_stitcherView->width();
	double winH = m_stitcherView->height();
	double px, py;
	window2pano_1x1(panoW, panoH, winW, winH, px, py, iPosX, iPosY, isStereo());
	float theta, phi;
	py = 1.0f - py;
	XYnToThetaPhi(px, py, theta, phi);
	vec3 u;
	sphericalToCartesian(theta, phi, u);

// 	QPointF dstPos = m_stitcherView->boundingRect().center();
// 	window2pano_1x1(panoW, panoH, winW, winH, px, py, dstPos.x(), dstPos.y(), isStereo());
// 	py = 1.0f - py;
// 	XYnToThetaPhi(px, py, theta, phi);
// 	sphericalToCartesian(theta, phi, v);

	mat3 mOrg = mat3_id, mRot = mat3_id, m = mat3_id;
	mRot.set_rot(u, vec3_z);

	u = vec3(setting.m_fRoll * sd_to_rad, setting.m_fPitch * sd_to_rad, setting.m_fYaw * sd_to_rad);
	mOrg.set_rot_zxy(u);

	m = mult(mRot, mOrg);

	m.get_rot_zxy(u);

	setting.m_fYaw = u[2] * sd_to_deg;
	setting.m_fPitch = u[1] * sd_to_deg;
	setting.m_fRoll = u[0] * sd_to_deg;

	sharedImageBuffer->getStitcher()->restitch();

// 	QPoint pos(iPosX, iPosY);
// 	if (m_stitcherView){
// 		m_stitcherView->onMousePress(pos);
// 		QPointF dstPos = m_stitcherView->boundingRect().center();
// 		m_stitcherView->onMouseMove(QPoint(dstPos.x(), dstPos.y()));
// 	}	
}

void QmlMainWindow::onPressedInteractive(int iPosX, int iPosY)
{
	QPoint pos(iPosX, iPosY);
	if (m_interactView)
		m_interactView->onMousePress(pos);
}

void QmlMainWindow::onMovedInteractive(int iPosX, int iPosY)
{
	QPoint pos(iPosX, iPosY);
	if (m_interactView)
		m_interactView->onMouseMove(pos);
}

void QmlMainWindow::onReleasedInteractive(int iPosX, int iPosY)
{
	QPoint pos(iPosX, iPosY);
	if (m_interactView)
		m_interactView->onMouseRelease(pos);
}

void QmlMainWindow::snapshotFrame()
{
	GlobalAnimSettings::CameraSettingsList& cameraSettingsList = m_d360Data.getGlobalAnimSettings().cameraSettingsList();

	PANO_LOG(QString("Snapshot current frame. (%1)").arg(CUR_TIME_H));
	QDir snapshotDir(sharedImageBuffer->getGlobalAnimSettings()->m_snapshotDir);
	if (!snapshotDir.exists()) {
		PANO_LOG(QString("Snapshot directory not exist! (%1)").arg(sharedImageBuffer->getGlobalAnimSettings()->m_snapshotDir));
		return;
	}
	for (int i = 0; i < cameraModuleMap.size(); i++)
	{
		cameraModuleMap.values()[i]->snapshot();
	}
}

void QmlMainWindow::snapshotPanoramaFrame()
{	
	PANO_LOG(QString("Snapshot current spherical frame. (%1)").arg(CUR_TIME_H));
	QDir snapshotDir(sharedImageBuffer->getGlobalAnimSettings()->m_snapshotDir);
	if (!snapshotDir.exists()) {
		PANO_LOG(QString("Panorama Snapshot directory not exist! (%1)").arg(sharedImageBuffer->getGlobalAnimSettings()->m_snapshotDir));
		return;
	}
	
	sharedImageBuffer->getStitcher()->snapshot();
}

QString QmlMainWindow::onImageFileDlg(QString filePath)
{
	if (!QDir(filePath).exists() || QDir(filePath).count() == 0) 
		return "";
	
	m_iPathList = QDir(filePath).entryList(QDir::NoDotAndDotDot | QDir::Dirs);

	if (m_iPathList.size() == 0)
		return "";

	return m_iPathList.join(",") + ",";
}

int QmlMainWindow::reloadCameraCalibrationFile(QString strCalibFile)
{
	if (strCalibFile == NULL)
		strCalibFile = m_d360Data.getCameraCalib();
	QFile file(strCalibFile);
	if (file.exists() && m_d360Data.parseCameraCalibrationFile(strCalibFile, m_d360Data.getGlobalAnimSettings().cameraSettingsList()) == 0)
	{
		sharedImageBuffer->getStitcher()->restitch(true);
		PANO_N_LOG("Reloaded the calibration file successfully.");
	}		
	else
	{
		PANO_N_WARN("Error in reloading the calibration file.");
	}		
	return 0;
}

bool QmlMainWindow::calibrate(int stepIndex)
{
	if (getGlobalAnimSetting().cameraSettingsList().size() < 2) {
		if (m_isStarted) {
			PANO_N_WARN("You don't have to calibrate if ONE camera.");
		}
		m_isStarted = true;
		emit started(true);			// Sending started signal to QML.		
		emit calibratingFinished();
		return false;
	}

	if (!QFile::exists(CALIB_WORK_PATH)) {
		if (m_d360Data.getGlobalAnimSettings().m_cameraCalibFile.isEmpty()) {
			m_isStarted = true;
			emit started(true);			// Sending started signal to QML.
		}
		emit calibratingFinished();
		PANO_N_ERROR("'PanoramaTools' directory not exist!");
		return false;
	}

	int ret = false;
	static int step = 0;

	if (stepIndex > -1)	step = stepIndex;

	int fov = m_d360Data.getGlobalAnimSettings().m_fov;
	int lensType = m_d360Data.getGlobalAnimSettings().m_lensType == 1 ? 3 : m_d360Data.getGlobalAnimSettings().m_lensType;

	emit loadingMessageReceived("Calibrating...");

	switch (step++)
	{
	case 0:
		ret = m_calib->calibrate(QString("pto_gen -o project.pto -p %1 -f %2 *.jpg").ARGN(lensType).ARGN(fov));
		break;
	case 1:
		ret = m_calib->calibrate("cpfind -o cpoints.pto project.pto");
		break;
	case 2:
		ret = m_calib->calibrate("cpclean -o pre_calib.pto cpoints.pto");
		break;
	case 3:
		ret = m_calib->calibrate("pto_var --opt y,p,r -o pre_step1.pto pre_calib.pto");
		break;
	case 4:
		ret = m_calib->calibrate("autooptimiser -n -o post_step1.pto pre_step1.pto");
		break;
	case 5:
		ret = m_calib->calibrate("pto_var --opt y,p,r,v -o pre_step2.pto post_step1.pto");
		break;
	case 6:
		ret = m_calib->calibrate("autooptimiser -n -o post_step2.pto pre_step2.pto");
		break;
	case 7:
		ret = m_calib->calibrate("pto_var --opt y,p,r,v,b -o pre_step3.pto post_step2.pto");
		break;
	case 8:
		ret = m_calib->calibrate("autooptimiser -n -o post_step3.pto pre_step3.pto");
		break;
	case 9:
		ret = m_calib->calibrate("pto_var --opt y,p,r,v,a,b,c,d,e -o pre_step4.pto post_step3.pto");
		break;
	case 10:
		ret = m_calib->calibrate("autooptimiser -n -o calib_result.pto pre_step4.pto");
		break;
	case 11:
		ret = m_calib->calibrate("pto_var --opt Eev,!Eev0 -o pre_photometric.pto calib_result.pto");
		break;
	case 12:
		ret = m_calib->calibrate("vig_optimize -o complete.pto pre_photometric.pto", true);	// Final command
		break;
	default:
		ret = true;
		break;
	}

	if (ret && step == 100) {
		// Initialize PAC and PTS map
		QMap<QString, QString>  pacPTO;
		pacPTO["lensType"] = "f";
		pacPTO["yaw"] = "y";
		pacPTO["pitch"] = "p";
		pacPTO["roll"] = "r";
		pacPTO["fov"] = "v";
		pacPTO["k1"] = "a";
		pacPTO["k2"] = "b";
		pacPTO["k3"] = "c";
		pacPTO["offset_x"] = "d";
		pacPTO["offset_y"] = "e";
		pacPTO["expOffset"] = "Eev";

		QMap<QString, PTOVAL> paramMap = m_calib->getCalibParams();

		if (paramMap.size() == 0) {
			// Error Process
			if (m_d360Data.getGlobalAnimSettings().m_cameraCalibFile.isEmpty()) {
				m_isStarted = true;
				emit started(true);			// Sending started signal to QML.
			}
			m_calibResult = false;
			emit calibratingFinished();
			PANO_N_ERROR("Calibrating failed!");
			return false;
		}

		GlobalAnimSettings::CameraSettingsList& cameraSettings = m_d360Data.getGlobalAnimSettings().cameraSettingsList();

		int devIndex = 0;
		while (paramMap.contains(QString("%1%2").arg(IMAGE_VAL).arg(QString::number(devIndex)))) {
			PTOVAL entry = paramMap[QString("%1%2").arg(IMAGE_VAL).arg(QString::number(devIndex))];
			//qDebug() << endl << entry.value;
			QMap<QString, float> params = entry.camParams;
			QMapIterator<QString, QString> i(pacPTO);
			while (i.hasNext()) {
				i.next();
				QString key = i.value();
				float value = 0;
				if (params.contains(key)) {
					value = params[key];
				}
				if (i.key() == "lensType")
					cameraSettings[devIndex].m_cameraParams.m_lensType = (value == 3 ? CameraParameters::LensType_ptLens_Fullframe_Fisheye : (CameraParameters::LensType)(int)value);
				else if (i.key() == "yaw")
					cameraSettings[devIndex].m_cameraParams.m_yaw = value;
				else if (i.key() == "pitch")
					cameraSettings[devIndex].m_cameraParams.m_pitch = value;
				else if (i.key() == "roll")
					cameraSettings[devIndex].m_cameraParams.m_roll = value;
				else if (i.key() == "fov")
					cameraSettings[devIndex].m_cameraParams.m_fov = (value == 0 ? cameraSettings[0].m_cameraParams.m_fov : value);
				else if (i.key() == "k1")
					cameraSettings[devIndex].m_cameraParams.m_k1 = (value == 0 ? cameraSettings[0].m_cameraParams.m_k1 : value);
				else if (i.key() == "k2")
					cameraSettings[devIndex].m_cameraParams.m_k2 = (value == 0 ? cameraSettings[0].m_cameraParams.m_k2 : value);
				else if (i.key() == "k3")
					cameraSettings[devIndex].m_cameraParams.m_k3 = (value == 0 ? cameraSettings[0].m_cameraParams.m_k3 : value);
				else if (i.key() == "offset_x")
					cameraSettings[devIndex].m_cameraParams.m_offset_x = (value == 0 ? cameraSettings[0].m_cameraParams.m_offset_x : value);
				else if (i.key() == "offset_y")
					cameraSettings[devIndex].m_cameraParams.m_offset_y = (value == 0 ? cameraSettings[0].m_cameraParams.m_offset_y : value);
				else if (i.key() == "expOffset")
				{
					cameraSettings[devIndex].m_cameraParams.m_expOffset = value;
					cameraSettings[devIndex].exposure = value;
				}
				//qDebug() << QString("%1 (%2) = %3").arg(i.key()).arg(key).arg(value);
			}
			devIndex++;
			if (devIndex >= cameraSettings.size())
				break;
		}

		// Getting control points...
		m_cpList.clear();
		QMapIterator<QString, PTOVAL> iter(paramMap);
		while (iter.hasNext()) {
			iter.next();
			QString key = iter.key();
			PTOVAL value = iter.value();

			if (!key.startsWith(CP_VAL))	continue;

			QMap<QString, float> cpoint = value.camParams;
			CPOINT cp;
			cp.point1 = cpoint["n"];
			cp.point2 = cpoint["N"];
			cp.x1 = cpoint["x"];
			cp.y1 = cpoint["y"];
			cp.x2 = cpoint["X"];
			cp.y2 = cpoint["Y"];
			m_cpList.append(cp);
		}

		PANO_N_LOG("Calibrating final step finished.");
		if (m_d360Data.getGlobalAnimSettings().m_cameraCalibFile.isEmpty() && m_isStarted == false) {
			m_isStarted = true;
			emit started(true);			// Sending started signal to QML.
		}
		else {
			m_calibResult = true;
			emit calibratingFinished();
		}
		sharedImageBuffer->getStitcher()->restitch(true);
	}
	
	if (ret == false)
	{
		// Error Process
		if (m_d360Data.getGlobalAnimSettings().m_cameraCalibFile.isEmpty()) {
			m_isStarted = true;
			emit started(true);			// Sending started signal to QML.
		}
		m_calibResult = false;
		emit calibratingFinished();
		PANO_N_ERROR("Calibrating failed!");
	}

	return ret;
}

int QmlMainWindow::getCPointCount(int camIndex1, int camIndex2)
{
	if (camIndex1 == camIndex2)	return 0;

	if (camIndex1 < 0 || camIndex2 < 0)	return 0;

	QListIterator<CPOINT> iter(m_cpList);
	int count = 0;
	while (iter.hasNext()) {
		CPOINT cp = iter.next();

		if ((cp.point1 == camIndex1 && cp.point2 == camIndex2) ||
			(cp.point2 == camIndex1 && cp.point1 == camIndex2))
			count++;
	}

	return count;
}

QString QmlMainWindow::getCPoint(int index, int camIndex1, int camIndex2)
{
	if (camIndex1 == camIndex2)	return "";

	if (camIndex1 < 0 || camIndex2 < 0)	return "";

	if (index < 0)	return "";

	QListIterator<CPOINT> iter(m_cpList);
	QString cpStr = "";
	int selIndex = 0;
	while (iter.hasNext()) {
		CPOINT cp = iter.next();

		if (!((cp.point1 == camIndex1 && cp.point2 == camIndex2) ||
			(cp.point2 == camIndex1 && cp.point1 == camIndex2)))
			continue;

		if (selIndex == index) {
			cpStr = QString("%1:%2:%3:%4").ARGN(cp.x1).ARGN(cp.y1).ARGN(cp.x2).ARGN(cp.y2);
			break;
		}
		selIndex++;
	}

	return cpStr;
}

int QmlMainWindow::getCPointCountEx(int camIndex)
{
	if (camIndex < 0)	return 0;

	QListIterator<CPOINT> iter(m_cpList);
	int count = 0;
	while (iter.hasNext()) {
		CPOINT cp = iter.next();

		if (cp.point1 == camIndex || cp.point2 == camIndex)
			count++;
	}

	return count;
}

QString QmlMainWindow::getCPointEx(int index, int camIndex)
{
	if (camIndex < 0)	return "";

	if (index < 0)	return "";

	QMap<QString, int> cpGroupMap;
	int groupIndex = 0;
	QListIterator<CPOINT> iter(m_cpList);
	while (iter.hasNext()) {
		CPOINT cp = iter.next();
		QString cpGroup = QString("%1:%2").ARGN(cp.point1).ARGN(cp.point2);
		if (cpGroupMap.contains(cpGroup))	continue;
		else
			cpGroupMap[cpGroup] = groupIndex++;
	}

	QString cpStr = "";
	int selIndex = 0;
	iter.toFront();
	while (iter.hasNext()) {
		CPOINT cp = iter.next();

		if (!(cp.point1 == camIndex || cp.point2 == camIndex))
			continue;

		if (selIndex == index) {
			QString cpGroup = QString("%1:%2").ARGN(cp.point1).ARGN(cp.point2);
			if (cp.point1 == camIndex)
				cpStr = QString("%1:%2:%3").ARGN(cp.x1).ARGN(cp.y1).ARGN(cpGroupMap[cpGroup]);
			else
				cpStr = QString("%1:%2:%3").ARGN(cp.x2).ARGN(cp.y2).ARGN(cpGroupMap[cpGroup]);
			break;
		}
		selIndex++;
	}

	return cpStr;
}

void QmlMainWindow::finishedSnapshot(int deviceIndex)
{
	PANO_DLOG(QString("Snapshot image saved. [%1]").ARGN(deviceIndex));
	snapshotCount++;
	
	if (snapshotCount != m_d360Data.getGlobalAnimSettings().cameraSettingsList().size())
		return;

	if (m_connectedCameras == false)
	{
		PANO_N_LOG("Project started!");
		m_isStarted = true;
		m_isAboutToStop = false;
		m_isClose = false;
		emit started(true);			// Sending started signal to QML.
	}
	//else if (m_d360Data.getGlobalAnimSettings().m_cameraCalibFile.isEmpty() && m_connectedCameras == false)
	//	calibrate(0);
	else if (m_connectedCameras == true)
		calibrate(0);

	m_connectedCameras = true;
	m_isEndCaptureThreads = false;
}

void QmlMainWindow::finishedSphericalSnapshot()
{
	PANO_DLOG(QString("Spherical Snapshot image saved."));	

	// should send the signal of snapshot finished to QML
}

void QmlMainWindow::startSingleCameraCalibration()
{
	if (!m_selfCalib)
	{
		m_selfCalib = new SingleCameraCalibProcess();
		m_selfCalib->setSharedImageBuffer(sharedImageBuffer);
		m_selfCalib->initialize();
	}
	
	int selectedCameraIndex = 0;
	m_selfCalib->setParameter(
		selectedCameraIndex,
		CameraParameters::LensType_opencvLens_Standard,
		CHECKBOARD_WIDTH, CHECKBOARD_HEIGHT,
		10);
	sharedImageBuffer->getGlobalAnimSettings()->m_stitch = false;
	sharedImageBuffer->setLiveGrabber(selectedCameraIndex);
}

void QmlMainWindow::startWithLiveGrabber(int selectedCameraIndex)
{
	sharedImageBuffer->setLiveGrabber(selectedCameraIndex);
}

void QmlMainWindow::takeSnapshot()
{

	//m_selfCalib->takeSnapshot();
}

void QmlMainWindow::luTChanged(const QVariantList &lutList, int colorType)
{
	printf("LuT value changed: ");
	for (int i = 0; i < lutList.size(); i++)
		printf("%f ", lutList[i].toFloat());
	printf("\n");

	sharedImageBuffer->getGlobalAnimSettings()->setLuTData((QVariantList &)lutList, colorType);
	sharedImageBuffer->getStitcher()->restitch();
}

bool QmlMainWindow::singleCalibrate()
{
	return m_selfCalib->calibrate();
}

void QmlMainWindow::finishSingleCameraCalibration()
{
	if (sharedImageBuffer->getGlobalAnimSettings() == NULL)
		return;

	sharedImageBuffer->getGlobalAnimSettings()->m_stitch = true;
	sharedImageBuffer->setLiveGrabber(-1);
	if (m_selfCalib)
	{
		delete m_selfCalib;
		m_selfCalib = NULL;
	}
}

void QmlMainWindow::startSingleCapture()
{
	if (m_selfCalib) m_selfCalib->startCapture();
}

void QmlMainWindow::stopSingleCapture()
{
	if (m_selfCalib) m_selfCalib->stopCapture();
}

void QmlMainWindow::applySingle()
{
	if (!m_selfCalib) return;
	m_selfCalib->apply();
}

void QmlMainWindow::startSingleCalibrating()
{
	if (m_selfCalib) m_selfCalib->calibrate();
}

void QmlMainWindow::startCalibrating()
{
	m_calib->initialize();
	snapshotCount = 0;
	for (int i = 0; i < cameraModuleMap.size(); i++)
	{
		cameraModuleMap.values()[i]->snapshot(true);
	}
}

void QmlMainWindow::setSingleParams(int camIndex, int lensTypeIndex,
	int boardSizeW, int boardSizeH, int snapshotNumber)
{
	CameraParameters::LensType lensType = lensTypeIndex == 0 ?
		CameraParameters::LensType_opencvLens_Standard :
		CameraParameters::LensType_opencvLens_Fisheye;
	if (m_selfCalib) m_selfCalib->setParameter(camIndex, lensType, boardSizeW, boardSizeH, snapshotNumber);
}

bool QmlMainWindow::isLeftEye(int iDeviceNum)
{
	std::vector< int >& indices = m_d360Data.getGlobalAnimSettings().getLeftIndices();
	for (int i = 0; i < indices.size(); i++)
	{
		if (indices[i] == iDeviceNum)
			return true;
	}
	return false;
}

bool QmlMainWindow::isRightEye(int iDeviceNum)
{
	std::vector< int >& indices = m_d360Data.getGlobalAnimSettings().getRightIndices();
	for (int i = 0; i < indices.size(); i++)
	{
		if (indices[i] == iDeviceNum)
			return true;
	}
	return false;
}

QList<qreal> QmlMainWindow::getSeamLabelPos(int camIndex)
{
	QList<qreal> seamPos;
	seamPos.push_back(-1); seamPos.push_back(-1);
	if (camIndex < 0 || camIndex >= m_d360Data.getGlobalAnimSettings().cameraSettingsList().size())
		return seamPos;

	GlobalAnimSettings& setting = g_mainWindow->getGlobalAnimSetting();
	
	CameraParameters& cam = setting.getCameraInput(camIndex).m_cameraParams;
	mat3 m = getCameraViewMatrix(cam.m_yaw, cam.m_pitch, cam.m_roll);
	mat3 inverseMatrix;
	invert(inverseMatrix, m);

	mat3 mPlacement = mat3_id;
	vec3 u(setting.m_fRoll * sd_to_rad, setting.m_fPitch * sd_to_rad, setting.m_fYaw * sd_to_rad);
	mPlacement.set_rot_zxy(u);
	
	mat3 finalMat;
	mult(finalMat, mPlacement, inverseMatrix);

	vec3 cartesian = finalMat * vec3_z;

	float theta, phi;
	cartesianTospherical(cartesian, theta, phi);
	float x_n, y_n;
	ThetaPhiToXYn(theta, phi, x_n, y_n);

	seamPos[0] = g_panoramaWidth * x_n;
	seamPos[1] = g_panoramaHeight * (1 - y_n);
	return seamPos;
}

bool QmlMainWindow::enableSeam(int camIndex1, int camIndex2)
{
	if (!m_isStarted || m_isAboutToStop)	return false;

	sharedImageBuffer->selectView(camIndex1, camIndex2);

	return true;
}

void QmlMainWindow::onNotify(int type, QString title, QString msg)
{
	QMap<int, QString> titleMap;
	titleMap[PANO_LOG_LEVEL::CRITICAL] = "Error";
	titleMap[PANO_LOG_LEVEL::WARNING] = "Warning";
	titleMap[PANO_LOG_LEVEL::INFO] = "Information";

	QString titleStr = "";

	if (!(title.isNull() || title.isEmpty()))
		titleStr = title;
	else if (type >= PANO_LOG_LEVEL::WARNING && type <= PANO_LOG_LEVEL::CRITICAL)
		titleStr = titleMap[type];

	QString msgStr = "";
	if (!(msg.isNull() || msg.isEmpty()))
		msgStr = msg;

	//m_notifyMsg = titleStr + ":" + msgStr;
	QString timeStr = QTime::currentTime().toString("hh:mm");
	m_notifyMsg = titleStr + ":" + msgStr + ":" + timeStr;
	
	m_notifyList.append(m_notifyMsg);
	emit notify();
}

int QmlMainWindow::getNotificationCount()
{
	QListIterator<QString> iter(m_notifyList);
	int count = 0;
	while (iter.hasNext()) {
		QString notifyMsg = iter.next();
		count++;
	}

	return count;
}

QString QmlMainWindow::getNotification(int index)
{
	QListIterator<QString> iter(m_notifyList);
	QString notifyStr = "";
	//if (m_notifyList.size() == 0) return "";
	notifyStr = m_notifyList.at(index);

	return notifyStr;
}

bool QmlMainWindow::removeNotification(int index)
{
	QListIterator<QString> iter(m_notifyList);
	while (iter.hasNext())
	{
		iter.next();
		if (index == -1)
			m_notifyList.clear();
		else{
			m_notifyList.removeAt(index);
		}
		return true;
	}
	       
	return false;
	
}

QList<qreal> QmlMainWindow::getSlotInfo(QString name, QString ext, int type)
{
	QList<qreal> slotInfoList;	

	if (type == D360::Capture::CAPTURE_DSHOW)
	{
		// get width, height, fps
		int width, height, fps;

		CaptureDevices *capDev = new CaptureDevices();
        //std::string strDevicePath = m_videoDevices[0].path.toUtf8().constData();
		std::string strDevicePath = m_videoDevices[0].path;
		capDev->getVideoDeviceInfo(strDevicePath, width, height, fps);
		slotInfoList << width << height << fps;
		delete capDev;

		return slotInfoList;
	}


	SlotInfo slot(this, type);

	QString path = name;
	if (type == D360::Capture::CAPTURE_FILE) {
		QDir curPath(name + "/");
		QStringList fileList = curPath.entryList(QDir::Files);
		foreach(const QString file, fileList) {
			if (file.endsWith("." + ext)) {
				path += "/" + file;
				break;
			}
		}
	}

	if (m_connectedCameras) {
		for (unsigned i = 0; i < m_d360Data.getGlobalAnimSettings().cameraSettingsList().size(); i++)
		{
			CameraInput cameraInput = m_d360Data.getGlobalAnimSettings().getCameraInput(i);

			QString strCameraInputName = cameraInput.name;
			if (type == D360::Capture::CAPTURE_FILE) {
				strCameraInputName = cameraInput.fileDir;
			}

			if (strCameraInputName == name)
			{
				slotInfoList << cameraInput.xres << cameraInput.yres << cameraInput.fps;
				return slotInfoList;
			}
		}
	}

	if (slot.open(path) < 0)
	{
		PANO_N_ERROR(QString("Can not open slot (%1)").arg(name));
		slotInfoList << 640 << 480 << 30;
		return slotInfoList;
	}
	slotInfoList << slot.getWidth() << slot.getHeight() << slot.getRate();

	return slotInfoList;
}

bool QmlMainWindow::addBanner(int windowWidth, int windowHeight, QPoint pt0, QPoint pt1, QPoint pt2, QPoint pt3, QString bannerFile, bool isVideo)
{
	// conversion from window-points to gl-coords(1-1-1)
	QPoint pt[4] = { pt0, pt1, pt2, pt3 };
	double px, py;
	double panoW = getPanoXres();
	double panoH = getPanoYres();
	vec2 quad[4];
	bool isStereoRight = false;

	for (int i = 0; i < 4; i++)
	{
		bool atRightEye = window2pano_1x1(panoW, panoH, windowWidth, windowHeight, px, py, pt[i].x(), pt[i].y(), isStereo());
		if (i == 0)
			isStereoRight = atRightEye;
		quad[i].x = px;
		quad[i].y = 1 - py;
	}

	std::shared_ptr<D360Stitcher> stitcher = sharedImageBuffer->getStitcher();
	addBanner(quad, bannerFile, isVideo, isStereoRight);
	stitcher->restitch();

	return true;
}

bool QmlMainWindow::addBanner(vec2 quad[], QString bannerFile, bool isVideo, bool isStereoRight)
{
	// load file
	SlotInfo slot(this);
	QImage image;

	if (isVideo)
	{
		if (slot.open(bannerFile) < 0)
		{
			PANO_N_ERROR(QString("Can not open Banner video file (%1)").arg(bannerFile));
			return false;
		}
	}
	else
	{
		QString ext = bannerFile.right(bannerFile.length() - bannerFile.lastIndexOf(".") - 1);
		if (!image.load(bannerFile, ext.toLocal8Bit().data()))
		{
			PANO_N_ERROR(QString("Can not open Banner image file (%1)").arg(bannerFile));
			return false;
		}
	}

	// global setting
	GlobalAnimSettings& settings = m_d360Data.getGlobalAnimSettings();

	// banner info
	std::shared_ptr<D360Stitcher> stitcher = sharedImageBuffer->getStitcher();
	stitcher->lockBannerMutex();

	BannerInfo& banner = stitcher->createNewBanner();
	banner.filePath = bannerFile;
	banner.isVideo = isVideo;
	banner.isStereoRight = isStereoRight;
	memcpy(banner.quad, quad, sizeof(banner.quad));

	// calculate homography for banner
	GlobalAnimSettings& setting = g_mainWindow->getGlobalAnimSetting();
	vec3 zxy(setting.m_fRoll*sd_to_rad, setting.m_fPitch*sd_to_rad, setting.m_fYaw*sd_to_rad);
	mat3 m_ = mat3_id;
	m_.set_rot_zxy(zxy);
	mat3 m = mat3_id;
	invert(m, m_);

	vec3 paiV[4];
	for (int pi = 0; pi < 4; pi++)
	{
		int qi = (pi + 3) % 4;

		float theta, phi;
		XYnToThetaPhi(banner.quad[pi].x, banner.quad[pi].y, theta, phi);
		vec3 xyz;
		sphericalToCartesian(theta, phi, xyz);

		paiV[qi] = m * xyz;
	}
	const vec3& paiOrg = paiV[0]; // org
	vec3 ux = paiV[1] - paiOrg;
	vec3 uy = paiV[3] - paiOrg;
	vec3 uz;
	cross(uz, ux, uy);
	cross(uy, uz, ux);

	ux.normalize();
	uy.normalize();
	uz.normalize();
	banner.paiPlane.set_col(0, ux);
	banner.paiPlane.set_col(1, uy);
	banner.paiPlane.set_col(2, uz);
	banner.paiPlane.set_col(3, paiOrg);

	dot(banner.paiZdotOrg, uz, paiOrg);

	vector<vec2> paiPlane;
	for (int i = 0; i < 4; i++)
	{
		float zv;
		dot(zv, uz, paiV[i]);
		float t = banner.paiZdotOrg / zv;
		vec3 vp = t * paiV[i];
		vp -= paiOrg;
		float x;
		dot(x, vp, ux);
		float y;
		dot(y, vp, uy);

		paiPlane.push_back(vec2(x, y));
	}

	vector<vec2> bannerTex;
	bannerTex.push_back(vec2(0, 0));
	bannerTex.push_back(vec2(1, 0));
	bannerTex.push_back(vec2(1, 1));
	bannerTex.push_back(vec2(0, 1));

	banner.homography = findHomography(paiPlane, bannerTex);

	if (isVideo)
	{
		BannerThread* bannerThread = new BannerThread(sharedImageBuffer, banner.id, bannerFile, slot.getWidth(), slot.getHeight(), slot.getRate());
		PANO_CONN(bannerThread, started(), bannerThread, process());
		PANO_CONN(bannerThread, started(int, QString, int), this, startedThread(int, QString, int));
		PANO_CONN(bannerThread, finished(int, QString, int), this, finishedThread(int, QString, int));
		if (bannerThread->connect())
			bannerThread->start();
		m_bannerThreads.push_back(bannerThread);
	}
	else
	{
		stitcher->updateBannerImageFrame(image, banner.id);
	}

	stitcher->unlockBannerMutex();

	return true;
}

void QmlMainWindow::removeAllBanners()
{
	for (int i = 0; i < m_bannerThreads.size(); i++)
	{
		BannerThread* banner = m_bannerThreads[i];
		banner->stopBannerThread(); // will be deleted on finishedThread().
		if (banner->isRunning())
		{
			banner->terminate();
			banner->wait();
		}
		delete banner;
		banner = NULL;
	}
	m_bannerThreads.clear();

	std::shared_ptr<D360Stitcher> stitcher = sharedImageBuffer->getStitcher();
	stitcher->lockBannerMutex();
	stitcher->cueRemoveBannerAll();
	stitcher->unlockBannerMutex();
	stitcher->restitch();
}

void QmlMainWindow::removeLastBanner()
{
	std::shared_ptr<D360Stitcher> stitcher = sharedImageBuffer->getStitcher();
	stitcher->lockBannerMutex();
	BannerInfo* bannerInfo = stitcher->getBannerLast();
	if (bannerInfo)
	{
		if (bannerInfo->isVideo)
		{
			for (int i = 0; i < m_bannerThreads.size(); i++)
			{
				BannerThread* banner = m_bannerThreads[i];
				if (banner->getBannerId() == bannerInfo->id)
				{
					// will be deleted on finishedThread().
					banner->stopBannerThread();
					if (banner->isRunning())
					{
						banner->terminate();
						banner->wait();
					}
					delete banner;
					banner = NULL;
					m_bannerThreads.erase(m_bannerThreads.begin() + i);
					break;
				}
			}
		}
		stitcher->cueRemoveBannerLast();
	}
	stitcher->unlockBannerMutex();
	stitcher->restitch();
}

void QmlMainWindow::removeBannerAtIndex(int index)
{
	std::shared_ptr<D360Stitcher> stitcher = sharedImageBuffer->getStitcher();
	stitcher->lockBannerMutex();
	BannerInfo* bannerInfo = stitcher->getBannerAtIndex(index);
	if (bannerInfo)
	{
		if (bannerInfo->isVideo)
		{
			for (int i = 0; i < m_bannerThreads.size(); i++)
			{
				BannerThread* banner = m_bannerThreads[i];
				if (banner->getBannerId() == bannerInfo->id)
				{
					// will be deleted on finishedThread().
					banner->stopBannerThread();
					if (banner->isRunning())
					{
						banner->terminate();
						banner->wait();
					}
					delete banner;
					banner = NULL;
					m_bannerThreads.erase(m_bannerThreads.begin() + i);
					break;
				}
			}
		}
		stitcher->cueRemoveBannerAtIndex(index);
	}
	stitcher->unlockBannerMutex();
	stitcher->restitch();
}

QString QmlMainWindow::getPosWindow2Pano(int w, int h, int x, int y){
	int width = m_d360Data.getGlobalAnimSettings().getPanoXres();
	int height = m_d360Data.getGlobalAnimSettings().getPanoYres();

	// mapping to width x height coordinate
	double px, py;
	window2pano_1x1(width, height, w, h, px, py, x, y, isStereo());
	int mappedX = width * px;
	int mappedY = height * py;

	//if (mappedX > width || mappedY > height)
	//	return;
	QString posStr = QString("%1:%2").ARGN(mappedX).ARGN(mappedY);
	return posStr;
}

QString QmlMainWindow::getPosPano2Window(int w, int h, int px, int py){
	int width = m_d360Data.getGlobalAnimSettings().getPanoXres();
	int height = m_d360Data.getGlobalAnimSettings().getPanoYres();

	// mapping to width x height coordinate
	double x, y;
	pano2window(width, height, w, h, px, py, x, y, isStereo(),false);
	int mappedX = x;
	int mappedY = y;

	//if (mappedX > width || mappedY > height)
	//return;
	QString posStr = QString("%1:%2").ARGN(mappedX).ARGN(mappedY);
	return posStr;

}

void QmlMainWindow::resetCamSettings()
{
	GlobalAnimSettings& setting = g_mainWindow->getGlobalAnimSetting();
	setting.m_fYaw = 0;
	setting.m_fPitch = 0;
	setting.m_fRoll = 0;
	sharedImageBuffer->getStitcher()->restitch();
}

int QmlMainWindow::getScreenCount()
{
	/*
	foreach(QScreen *screen, QGuiApplication::screens()) {
		qDebug() << "Information for screen:" << screen->name();
		qDebug() << "  Available geometry:" << screen->availableGeometry().x() << screen->availableGeometry().y() << screen->availableGeometry().width() << "x" << screen->availableGeometry().height();
		qDebug() << "  Available size:" << screen->availableSize().width() << "x" << screen->availableSize().height();
		qDebug() << "  Available virtual geometry:" << screen->availableVirtualGeometry().x() << screen->availableVirtualGeometry().y() << screen->availableVirtualGeometry().width() << "x" << screen->availableVirtualGeometry().height();
		qDebug() << "  Available virtual size:" << screen->availableVirtualSize().width() << "x" << screen->availableVirtualSize().height();
		qDebug() << "  Depth:" << screen->depth() << "bits";
		qDebug() << "  Geometry:" << screen->geometry().x() << screen->geometry().y() << screen->geometry().width() << "x" << screen->geometry().height();
		qDebug() << "  Logical DPI:" << screen->logicalDotsPerInch();
		qDebug() << "  Logical DPI X:" << screen->logicalDotsPerInchX();
		qDebug() << "  Logical DPI Y:" << screen->logicalDotsPerInchY();		
		qDebug() << "  Physical DPI:" << screen->physicalDotsPerInch();
		qDebug() << "  Physical DPI X:" << screen->physicalDotsPerInchX();
		qDebug() << "  Physical DPI Y:" << screen->physicalDotsPerInchY();
		qDebug() << "  Physical size:" << screen->physicalSize().width() << "x" << screen->physicalSize().height() << "mm";		
		qDebug() << "  Refresh rate:" << screen->refreshRate() << "Hz";
		qDebug() << "  Size:" << screen->size().width() << "x" << screen->size().height();
		qDebug() << "  Virtual geometry:" << screen->virtualGeometry().x() << screen->virtualGeometry().y() << screen->virtualGeometry().width() << "x" << screen->virtualGeometry().height();
		qDebug() << "  Virtual size:" << screen->virtualSize().width() << "x" << screen->virtualSize().height();
	}
	*/

	return QGuiApplication::screens().size();	
}

QString QmlMainWindow::getFullScreenInfoStr(int screenNo)
{		
	QRect rect = getFullScreenInfo(screenNo);
	QString strRect = QString::number(rect.x()) + ":" + QString::number(rect.y()) + ":" + QString::number(rect.width()) + ":" + QString::number(rect.height());
	return strRect;
}

QRect QmlMainWindow::getFullScreenInfo(int screenNo)
{
	QScreen *screen = QGuiApplication::screens()[screenNo];
	return screen->geometry();	
}

QString QmlMainWindow::getWeightMapFile(int cameraIndex)
{
	QString fileName = QString::number(cameraIndex) + "map.png";
	int width = 600, height = 600;

	QImage image(width, height, QImage::Format_ARGB32);
	image.fill(Qt::white);
	
	QRgb value;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			int weight = rand() % 255;
			weight = 128;
			value = qRgba(0, 0, 0, weight);					
			image.setPixel(i, j, value);			
		}
	}	
	
	//image.save(fileName, NULL, 100);	

	setUpdatedWeightMapFile("1map.aaa");

	return fileName;
}

void QmlMainWindow::setUpdatedWeightMapFile(QString strFileName)
{	
	// link this file to INI
	QImage image;
	image.load(strFileName, "png");
	
	int width = image.width();
	int height = image.height();
	int* weightMap = new int[width * height];
		
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{	
			// This value is the ONLY painted value by user (Grading value)
			QRgb rgbValue = image.pixel(i, j);		
			//int red = qRed(rgbValue);			
			//int alpha = -(red - 256) / 2.56;	
			int alpha = qAlpha(rgbValue);
			weightMap[i*width + j] = alpha;			
		}
	}
}

void QmlMainWindow::setWeightMapEditMode(bool isEditMode)
{
	sharedImageBuffer->setWeightMapEditEnabled(isEditMode);
}

void QmlMainWindow::setDrawWeightMapSetting(int cameraIndex, int cameraIndex_, int radius, float strength, float fallOff, bool isIncrement, int nEyeMode)
{		
	WeightMapEyeMode eyeMode = (WeightMapEyeMode)nEyeMode;

	m_weightMap_radius = radius;	
	m_weightMap_strength = strength / 100.f;
	m_weightMap_fallOff = fallOff / 100.f;
	m_weightMap_isIncrement = isIncrement;
	

	// calculate the ratio of all setting values comparing with 320x240 screen
	int width = m_d360Data.getGlobalAnimSettings().getPanoXres();
	int height = m_d360Data.getGlobalAnimSettings().getPanoYres();
	m_weightMap_radius *= width/1024.f;

	bool bCamIdxChanged = false;
	bCamIdxChanged =( m_weightMap_cameraIndex != cameraIndex || m_weightMap_cameraIndex_ != cameraIndex_) ? true : false;
	if (m_weightMap_cameraIndex != cameraIndex || m_weightMap_eyeMode != eyeMode || m_weightMap_cameraIndex_ != cameraIndex_){
		m_weightMap_cameraIndex = cameraIndex;
		m_weightMap_cameraIndex_ = cameraIndex_;
		m_weightMap_eyeMode = eyeMode;
		m_eyeMode = eyeMode;
	}

	m_process->getStitcherThread()->updateWeightMapParams(m_weightMap_eyeMode, m_weightMap_cameraIndex, m_weightMap_cameraIndex_,  m_weightMap_strength, m_weightMap_radius, m_weightMap_fallOff, m_weightMap_isIncrement, bCamIdxChanged);

	PANO_LOG(QString("WeightMap Setting: cameraIndex[%1], radius[%2], strength[%3], fallOff[%4], increment[%5]")
		.arg(m_weightMap_cameraIndex).arg(m_weightMap_radius).arg(m_weightMap_strength).arg(m_weightMap_fallOff).arg(m_weightMap_isIncrement));	
}

void QmlMainWindow::setWeightMapChanged(bool isChanged)
{
	m_process->getStitcherThread()->setWeightMapChanged(isChanged);
}

void QmlMainWindow::drawWeightMap(int w, int h, int x, int y)
{

	int width = m_d360Data.getGlobalAnimSettings().getPanoXres();
	int height = m_d360Data.getGlobalAnimSettings().getPanoYres();	
	
	// mapping to width x height coordinate
	double px, py;
	bool isRightPos = window2pano_1x1(width, height, w, h, px, py, x, y, isStereo());
	int mappedX = width * px;
	int mappedY = height * py;
	
	if (mappedX > width || mappedY > height)
		return;	

	//PANO_LOG(QString("drawWeightMap: (%1, %2) -> (%3, %4), %5 ").arg(x).arg(y).arg(mappedX).arg(mappedY).arg(isRightPos));

	m_process->getStitcherThread()->updateWeightMapParams(m_weightMap_eyeMode, m_weightMap_cameraIndex, m_weightMap_cameraIndex_, m_weightMap_strength, m_weightMap_radius, m_weightMap_fallOff, m_weightMap_isIncrement, false, mappedX, mappedY, isRightPos);
	setWeightMapChanged(true);

	//saveWeightMapToFile(weightMapBuffer, width, height);
}

void QmlMainWindow::saveWeightMapToFile(int *weightMap, int width, int height)
{	
	QImage image(width, height, QImage::Format_ARGB32);
	image.fill(Qt::white);

	QRgb value;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{						
			value = qRgba(weightMap[j*width + i], 0, 0, 255);
			image.setPixel(i, j, value);
		}
	}

	image.save("1.png", NULL, 100);		
}

void QmlMainWindow::resetWeightMap()
{
	m_process->getStitcherThread()->setWeightMapResetFlag(true);
	m_process->getStitcherThread()->restitch();

	sendWeightMapResetStatusChanged(true);
}

void QmlMainWindow::weightmapUndo()
{
	m_process->getStitcherThread()->weightmapUndo();
	m_process->getStitcherThread()->restitch();
}

void QmlMainWindow::weightmapRedo()
{
	m_process->getStitcherThread()->weightmapRedo();
	m_process->getStitcherThread()->restitch();
}

void QmlMainWindow::setWeightMapPaintingMode(int paintMode)
{
	WeightMapPaintMode _paintMode = (WeightMapPaintMode)paintMode;
	m_process->getStitcherThread()->setWeightmapPaintingMode(_paintMode);
}

// Session/Take functions
void QmlMainWindow::setSessionRootPath(QString dirPath)
{
	m_sessionRootPath = dirPath;
	if (m_sessionRootPath != "" && m_sessionRootPath[m_sessionRootPath.size() - 1] != '/')
		m_sessionRootPath = m_sessionRootPath + "/";

	m_d360Data.getGlobalAnimSettings().m_capturePath = m_sessionRootPath;

	QSettings settings("Centroid LAB", "Look3D");
	settings.beginGroup("TakeManagement");
	settings.setValue("SessionRootPath", m_sessionRootPath);
	settings.endGroup();
}

QString QmlMainWindow::getSessionRootPath()
{	
	QSettings settings("Centroid LAB", "Look3D");
	settings.beginGroup("TakeManagement");
	m_sessionRootPath = settings.value("SessionRootPath").toString();
	settings.endGroup();

	m_sessionRootPath = m_d360Data.getGlobalAnimSettings().m_capturePath;

	return m_sessionRootPath;
}

int QmlMainWindow::getLastSessionId()
{
	int tmp_sessionId = 0;
	QString sessionDate = QDate::currentDate().toString("yyyy-MM-dd_");	

	// Navigate all sub directories in sessionDirPath	
	QDir curPath(m_sessionRootPath);
	QStringList dirList = curPath.entryList(QDir::Dirs);
	foreach(const QString dirItem, dirList)
	{
		if (dirItem.startsWith(sessionDate))
		{
			QStringList sessionNameList = dirItem.split("_");
			if (sessionNameList.size() == 2)
			{
				int sessionId = sessionNameList.at(1).toInt();
				if (sessionId > tmp_sessionId)
					tmp_sessionId = sessionId;
			}
		}
	}

	return tmp_sessionId;
}

int QmlMainWindow::getLastTakeId(int sessionId)
{	
	// Navigate all sub directories in selected session path		
	QString sessionDirPath = generateSessionDirFullPath(sessionId);

	int tmp_takeId = 0;
	QDir curPath(sessionDirPath);
	QStringList fileList = curPath.entryList(QDir::Files);
	foreach(const QString fileItem, fileList) {
		if (fileItem.startsWith("Take_") && fileItem.endsWith(".mp4"))
		{			
			QString fileName = fileItem.left(fileItem.indexOf("."));
			QStringList fileNameList = fileName.split("_");
			if (fileNameList.size() == 2)
			{
				int takeId = fileNameList.at(1).toInt();
				if (takeId > tmp_takeId)
					tmp_takeId = takeId;
			}
		}
	}
	return tmp_takeId;
}

void QmlMainWindow::createNewSession()
{	
	// create session root directory if not exist.
	QDir sessionDir = QDir(m_sessionRootPath);
	if (!sessionDir.exists())
	{
		PANO_N_LOG("Capture directory doesn't exist.");
		CreateDirectory(m_sessionRootPath.toLocal8Bit(), NULL);
	}
		

	int sessionId = getLastSessionId() + 1;
	QString sessionName = generateSessionName(sessionId);
	bool ret = sessionDir.mkdir(sessionName);

	if (ret) {
		PANO_N_LOG("New session created successfully.");

		m_isNewConfiguration = false;

		m_qmlTakeManagement->insertSession(g_takeMgrModel, sessionName, m_sessionNameList.size());
		m_sessionNameList.push_back(sessionName);
		QStringList strList;
		strList.append("");
		m_CommentList.append(strList);
		m_takeNameList.clear();
	}
	else
		PANO_N_WARN("Failed to create new session.");

}

bool QmlMainWindow::isTakeNode(QModelIndex index)
{
	return m_qmlTakeManagement->isTakeNode(g_takeMgrModel, index);
}

void QmlMainWindow::initPlayback(QModelIndex index)
{
	resetPlayback(index);

	if (!connectPlayback())
	{
		PANO_N_ERROR(QString("Playback Connecting Failed!"));
	}
	else
	{
		PANO_LOG(QString("Connected Playback."));
	}
}

void QmlMainWindow::resetPlayback(QModelIndex index)
{
	m_curIndex = index;
	int takeId = index.row() + 1;
	QString takeFileName = QString("[Take_%1]").arg(convertNumber2String(takeId, 3));
	QVariant sessionData = index.parent().data(Qt::EditRole);
	QString sessionName = sessionData.toString();
	QString strLogFilePath = generateSessionLogFileFullPathFromData(sessionName);
	QFile logFile(strLogFilePath);
	if (logFile.exists() && logFile.open(QFile::ReadOnly))
	{
		QTextStream logStream(&logFile);
		QString logString = logStream.readAll();
		QStringList logStringList = logString.split("\r\n", QString::SkipEmptyParts);
		for (int i = 0; i < logStringList.size(); i++)
		{
			if (logStringList.at(i) == takeFileName)
			{
				if (logStringList.at(i + 1).startsWith("FilePath=") && logStringList.at(i + 1).endsWith(".mp4"))
					m_strTrackFilePath = logStringList.at(i + 1).mid(9);
				else
					m_strTrackFilePath = "";
// 				if (logStringList.at(i + 3).startsWith("Duration="))
// 					m_strDuration = logStringList.at(i + 3).mid(9);
// 				else
// 					m_strDuration = "";
				break;
			}
		}

		logFile.close();
	}

	int nPos = m_strTrackFilePath.lastIndexOf('/');
	QString filePath = m_strTrackFilePath.left(nPos + 1);
	QString fileName = m_strTrackFilePath.mid(nPos + 1);
	m_strImageFilePath = filePath + "weight" + fileName.left(fileName.length() - 3) + "jpg";
}

void QmlMainWindow::setDurationString(int nFrames)
{
	m_nDurationFrame = nFrames;
}

QString QmlMainWindow::getDurationString()
{
	struct ThreadStatisticsData statsData;
	statsData.nFramesProcessed = m_nDurationFrame + 1;
	QString strDuration = statsData.toString(m_d360Data.getFps(), 0);
	return strDuration;
}

int QmlMainWindow::getDuration()
{
	return m_nDurationFrame;
}

QString QmlMainWindow::getTakeComment(QModelIndex index)
{
	QString commentStr = "";
	if (isTakeNode(index))
	{
		QModelIndex parentIndex = index.parent();
		if (m_CommentList.size() > parentIndex.row() && m_CommentList.at(parentIndex.row()).size() > index.row() + 1)
			commentStr = m_CommentList.at(parentIndex.row()).at(index.row()+1);
	}
	return commentStr;
}

void QmlMainWindow::changeComment(QModelIndex index, QString strComment)
{
	if (!m_qmlTakeManagement->isTakeNode(g_takeMgrModel, index))
		return;
	QVariant sessionData = index.parent().data(Qt::EditRole);
	QString sessionName = sessionData.toString();
	int takeId = index.row()+1;
	QString strLogFilePath = generateSessionLogFileFullPathFromData(sessionName);
	QFile logFile(strLogFilePath);
	if (logFile.exists() && logFile.open(QFile::ReadOnly))
	{
		QTextStream logStream(&logFile);
		QString logString = logStream.readAll();
		QStringList logStringList = logString.split("\r\n", QString::KeepEmptyParts);
		QString oldComentString = logStringList.at(6 * takeId - 1);
		oldComentString.remove(8, oldComentString.size() - 8);
		QString newCommentStr = oldComentString.append(strComment);
		logStringList.replace(6 * takeId - 1, newCommentStr);

		logFile.close();
		if (logFile.open(QFile::WriteOnly | QFile::Truncate))
		{
			for (int i = 0; i < logStringList.size(); i++)
			{
				if (i == logStringList.size() - 1)
					logStream << logStringList.at(i);
				else
					logStream << logStringList.at(i) << "\r\n";
			}
			logFile.close();
		}
		QModelIndex parentIndex = index.parent();
		QStringList oldList = m_CommentList.at(parentIndex.row());
		oldList.replace(index.row() + 1, strComment);
		m_CommentList.replace(parentIndex.row(), oldList);
		//PANO_N_LOG(QString("%1 saved successfully.").arg(m_sessionTake.name));
	}
}

bool QmlMainWindow::startRecordTake()
{
	if (m_streamProcess)
	{
		PANO_N_LOG("Please stop previous streaming first.");
		return false;
	}
	

	struct ThreadStatisticsData statsData = sharedImageBuffer->getStitcher()->getStatisticsData();
	m_lastFramesProcessed = statsData.nFramesProcessed;

	int sessionId = getLastSessionId();	
	if (sessionId == 0 || m_isNewConfiguration)
	{
		createNewSession();
		sessionId = getLastSessionId();		
	}		

	QString sessionDirPath = generateSessionDirFullPath(sessionId);	
	int takeId = getLastTakeId(sessionId) + 1;
	QString takeFileName = QString("Take_%1").arg(convertNumber2String(takeId, 3));
	QString takeFileFullName = QString("%1/%2.mp4").arg(sessionDirPath).arg(takeFileName);
		

	m_sessionTake.name = takeFileName;
	m_sessionTake.filePath = takeFileFullName;
	m_sessionTake.startTime = QDateTime::currentDateTime().toString("yyyy.MM.dd hh:mm:ss:zzz");
	m_sessionTake.comment = QString("");
	
	// create streaming process for take.	
	m_streamProcess = new StreamProcess(sharedImageBuffer, true, this);

	bool retval = false;
	int videoCodec = m_d360Data.getGlobalAnimSettings().m_videoCodec;
	int audioCodec = m_d360Data.getGlobalAnimSettings().m_audioCodec;
	int audioLag = m_d360Data.getGlobalAnimSettings().m_audioLag;
	int sampleFmt = m_d360Data.getGlobalAnimSettings().m_sampleFmt;
	int panoWidth = m_d360Data.getGlobalAnimSettings().m_panoXRes;
	int panoHeight = m_d360Data.getGlobalAnimSettings().m_panoYRes;
	int channels = m_d360Data.getGlobalAnimSettings().getAudioChannelCount();
	int resultHeight = panoHeight;
	if (sharedImageBuffer->getGlobalAnimSettings()->isStereo())
		resultHeight *= 2;

	int startframe = m_d360Data.getGlobalAnimSettings().m_startFrame;
	int endframe = m_d360Data.getGlobalAnimSettings().m_endFrame;
	float fps = m_d360Data.getGlobalAnimSettings().m_fps;
	int crf = m_d360Data.getGlobalAnimSettings().m_crf;
	int nCameras = m_d360Data.getGlobalAnimSettings().cameraSettingsList().size();

	bool blLiveMode = m_d360Data.getGlobalAnimSettings().m_captureType == D360::Capture::CAPTURE_DSHOW;

	AudioInput * mic = NULL;
	if (blLiveMode)
	{
		if (!audioModuleMap.isEmpty())
			mic = audioModuleMap.first()->getMic();
	}
	else
	{
		mic = cameraModuleMap.first()->getAudioThread()->getMic();
	}

	int sampleRate = mic ? mic->getSampleRate() : 48000;
	retval = m_streamProcess->initialize(true, takeFileFullName, panoWidth, resultHeight,
		sharedImageBuffer->getGlobalAnimSettings()->m_sourcefps, channels,
		(AVSampleFormat)sampleFmt, sampleRate, sampleRate, audioLag, videoCodec, audioCodec, crf);

	if (retval == false)
	{
		PANO_N_ERROR("Cannot initialize output to HDD for creating the take!");
		delete m_streamProcess;
		m_streamProcess = NULL;
		return false;
	}		
	else
		sharedImageBuffer->setStreamer(m_streamProcess);		

	//QThread::msleep(300);

	QModelIndex curIndex = m_qmlTakeManagement->insertTake(g_takeMgrModel, m_sessionTake.name, m_sessionNameList.size() - 1, m_takeNameList.size());
	if (curIndex.isValid())	setCurIndex(curIndex);
	m_takeNameList.push_back(m_sessionTake.name);

	QString takeFileMaskName = QString("%1/weight%2.jpg").arg(sessionDirPath).arg(takeFileName);
	sharedImageBuffer->getStitcher()->recordMaskMap(takeFileMaskName);

	std::shared_ptr< D360Stitcher > stitcher = m_process->getStitcherThread();
	stitcher->buildOutputNode(panoWidth, resultHeight);

	return true;
}

bool QmlMainWindow::stopRecordTake(QString strComment)
{
	if (m_streamProcess && m_streamProcess->isTakeRecording())
	{
		m_streamProcess->stopStreamThread();

		//QThread::msleep(300);
		
		// Log this Take to the file
		QString strLogFilePath = generateSessionLogFileFullPath(getLastSessionId());
		QFile logFile(strLogFilePath);
		if (logFile.exists())
		{
			if (!logFile.open(QIODevice::Append | QIODevice::Text))
				return false;
		}
		else
		{
			if (!logFile.open(QIODevice::WriteOnly | QIODevice::Text))
				return false;
		}		
				
		if (strComment != NULL)
		{
			m_sessionTake.comment = strComment;
			m_CommentList.last().append(strComment);
		}
		else
			m_CommentList.last().append(QString(""));


		// calculate duration of sessionTake
		struct ThreadStatisticsData statsData = sharedImageBuffer->getStitcher()->getStatisticsData();
		m_sessionTake.duration = statsData.toString(m_d360Data.getFps(), m_lastFramesProcessed);
		
		QString strTake = QString("\n[%1]\nFilePath=%2\nStartTime=%3\nDuration=%4\nComment=%5\n").arg(m_sessionTake.name).arg(m_sessionTake.filePath).arg(m_sessionTake.startTime).arg(m_sessionTake.duration).arg(m_sessionTake.comment);
		logFile.write(strTake.toLocal8Bit(), strTake.length());
		logFile.close();

		PANO_N_LOG(QString("%1 saved successfully.").arg(m_sessionTake.name));						

		return true;
	}
	return false;
}

bool QmlMainWindow::startStreaming(QString rtmpAddress, int streamingWidth, int streamingHeight, int streamingMode)
{
	bool isWebRTC;
	if (streamingMode == RTMP)
		isWebRTC = false;
	else if (streamingMode == WEBRTC)
		isWebRTC = true;
	m_qmlApplicationSetting->setStreamingMode(isWebRTC);

	if (m_streamProcess && !isWebRTC)
	{
		emit loadingMessageReceived("StreamingFailed");
		PANO_N_LOG("Please stop previous streaming first.");
		return false;
	}

	m_lastFramesProcessed = sharedImageBuffer->getStitcher()->getStatisticsData().nFramesProcessed;

	// fill the values into sessionTake structure.
	struct ThreadStatisticsData statsData = sharedImageBuffer->getStitcher()->getStatisticsData();

	// create streaming process for take.	
	if (isWebRTC == true)
	{
		if (!m_streamProcess)
		{
			webRTC_StreamProcess* webRTC_StreamProcessPtr = new webRTC_StreamProcess(sharedImageBuffer, false, this);
			m_streamProcess = webRTC_StreamProcessPtr;
		}
		emit loadingMessageReceived("Creating webRTC stream...");
	}
	else
	{
		m_streamProcess = new StreamProcess(sharedImageBuffer, false, this);
		emit loadingMessageReceived("Creating RTMP stream...");
	}

	bool retval = false;
	int videoCodec = m_d360Data.getGlobalAnimSettings().m_videoCodec;
	int audioCodec = m_d360Data.getGlobalAnimSettings().m_audioCodec;
	int audioLag = m_d360Data.getGlobalAnimSettings().m_audioLag;
	int sampleFmt = m_d360Data.getGlobalAnimSettings().m_sampleFmt;
	int channels = m_d360Data.getGlobalAnimSettings().getAudioChannelCount();
	int resultHeight = streamingHeight;
	if (sharedImageBuffer->getGlobalAnimSettings()->isStereo())
		resultHeight *= 2;

	int nCameras = m_d360Data.getGlobalAnimSettings().cameraSettingsList().size();

	int crf = m_d360Data.getGlobalAnimSettings().m_crf;
	//bool blLiveMode = m_d360Data.getGlobalAnimSettings().m_captureType == D360::Capture::CAPTURE_DSHOW;
	bool blLiveMode = m_d360Data.getGlobalAnimSettings().m_captureType == D360::Capture::CAPTURE_PTGREY;

	AudioInput * mic = NULL;
	if (blLiveMode)
	{
		if (!audioModuleMap.isEmpty())
			mic = audioModuleMap.first()->getMic();
	}
	else
	{
		mic = cameraModuleMap.first()->getAudioThread()->getMic();
	}

	int sampleRate = mic ? mic->getSampleRate() : 48000;
	retval = m_streamProcess->initialize(true, rtmpAddress, streamingWidth, resultHeight,
		sharedImageBuffer->getGlobalAnimSettings()->m_sourcefps, channels,
		(AVSampleFormat)sampleFmt, sampleRate, sampleRate, audioLag, videoCodec, audioCodec, crf);

	if (retval == false)
	{
		emit loadingMessageReceived("StreamingFailed");
		PANO_N_ERROR("Cannot initialize rtmp streaming!");
		delete m_streamProcess;
		m_streamProcess = NULL;
		return false;
	}
	else
		sharedImageBuffer->setStreamer(m_streamProcess);

	std::shared_ptr< D360Stitcher > stitcher = m_process->getStitcherThread();
	stitcher->buildOutputNode(streamingWidth, resultHeight);

	//emit loadingMessageReceived("StreamingFailed");
	emit loadingMessageReceived("StreamingSuccessed");
	return true;
}

bool QmlMainWindow::stopStreaming()
{
	if (m_streamProcess && !m_streamProcess->isTakeRecording())
	{
		m_streamProcess->stopStreamThread();

		PANO_N_LOG(QString("Streaming stopped successfully."));

		return true;
	}
	return false;
}

void QmlMainWindow::setStreamingPath(QString rtmpAddress)
{
	m_d360Data.getGlobalAnimSettings().m_wowzaServer = rtmpAddress;
}

QString QmlMainWindow::getStreamingPath()
{
	return m_d360Data.getGlobalAnimSettings().m_wowzaServer;
}

void QmlMainWindow::deleteSession(QString strSessionPath)
{

}

void QmlMainWindow::deleteTake(QString strTakePath)
{

}

QString QmlMainWindow::travelSessionRootPath()
{		
	QString strTravelResult = "";

	// Navigate all sub directories in sessionRootDirPath	
	QDir rootPath(m_sessionRootPath);
	QStringList dirList = rootPath.entryList(QDir::Dirs);
	foreach(const QString dirItem, dirList)
	{
		if (dirItem == "." || dirItem == "..")
			continue;

		if (strTravelResult.length() == 0)
			strTravelResult = dirItem + ":";
		else
		{
			if (strTravelResult.endsWith(","))
				strTravelResult = strTravelResult.left(strTravelResult.lastIndexOf(","));
			strTravelResult += ";" + dirItem + ":";
		}

		QString strSessionPath = QString("%1/%2").arg(m_sessionRootPath).arg(dirItem);
		QDir sessionPath(strSessionPath);
		QStringList fileList = sessionPath.entryList(QDir::Files);
		foreach(const QString fileItem, fileList)
		{
			if (fileItem.startsWith("Take_") && fileItem.endsWith(".mp4"))
			{
				strTravelResult += fileItem + ",";
			}
		}
	}
	if (strTravelResult.endsWith(","))
		strTravelResult = strTravelResult.left(strTravelResult.lastIndexOf(","));
	return strTravelResult;
}

void QmlMainWindow::removeAllSessions(TakeMgrTreeModel* model,int position, int sessionCount)
{
	m_qmlTakeManagement->removeAllSession(model,position,sessionCount);
}


QObject* QmlMainWindow::takManagement() const
{
	return m_qmlTakeManagement;
}

QObject* QmlMainWindow::applicationSetting() const
{
	return m_qmlApplicationSetting;
}

void QmlMainWindow::setTakManagement(QObject *takManagement)
{
	if (m_qmlTakeManagement == takManagement)
		return;
	QmlTakeManagement *s = qobject_cast<QmlTakeManagement*>(takManagement);
	if (!s)
	{
		qDebug() << "Source must be a MCQmlTakeManagement type";
		return;
	}
	m_qmlTakeManagement = s;

	emit takManagementChanged(m_qmlTakeManagement);
}

void QmlMainWindow::setApplicationSetting(QObject* applicationSetting)
{
	if (m_qmlApplicationSetting == applicationSetting)
		return;

	QmlApplicationSetting* s = qobject_cast<QmlApplicationSetting*>(applicationSetting);
	if (!s)
	{
		qDebug() << "Source must be a MCQmlApplicationSetting type";
		return;
	}

	m_qmlApplicationSetting = s;

	emit applicationSettingChanged(m_qmlApplicationSetting);

}

void QmlMainWindow::setNodalVideoFilePath(int slotIndex, QString strVideoFilePath)
{
	m_d360Data.getGlobalAnimSettings().m_nodalVideoFilePathMap[slotIndex] = (strVideoFilePath == "Background footage video") ? "" : strVideoFilePath;
}

void QmlMainWindow::setNodalMaskImageFilePath(int slotIndex, QString strMaskImageFilePath)
{
	m_d360Data.getGlobalAnimSettings().m_nodalMaskImageFilePathMap[slotIndex] = (strMaskImageFilePath == "Background weight map") ? "" : strMaskImageFilePath;
	if (strMaskImageFilePath != "Background weight map")
		m_d360Data.getGlobalAnimSettings().m_haveNodalMaskImage = true;
}

QString QmlMainWindow::getNodalVideoFilePath(int slotIndex)
{
	QString strVideoFilePath;
	if (m_d360Data.getGlobalAnimSettings().m_nodalVideoFilePathMap.contains(slotIndex))
		strVideoFilePath = m_d360Data.getGlobalAnimSettings().m_nodalVideoFilePathMap[slotIndex];
	if (strVideoFilePath == NULL || strVideoFilePath == "")
	{
		strVideoFilePath = "Background footage video";
	}
	return strVideoFilePath;
}

QString QmlMainWindow::getNodalMaskImageFilePath(int slotIndex)
{ 
	QString strMaskImageFilePath = m_d360Data.getGlobalAnimSettings().m_nodalMaskImageFilePathMap[slotIndex];
	if (strMaskImageFilePath == NULL || strMaskImageFilePath == "")
	{
		strMaskImageFilePath = "Background weight map";
	}
	return strMaskImageFilePath;
}

void QmlMainWindow::setNodalCameraIndex(int index)
{
	m_d360Data.getGlobalAnimSettings().m_nodalVideoIndex = index; 

	if (index != -1) {
		m_d360Data.getGlobalAnimSettings().m_haveLiveBackground = true;
	}
}

// CT functions
void QmlMainWindow::setColorTemperature(int ctValue)
{	
	m_process->getStitcherThread()->setColorTemperature(ctValue);
	m_process->getStitcherThread()->calculateLightColorWithTemperature(ctValue);
}

void QmlMainWindow::newCalibFrame(unsigned char* buffer, int width, int height)
{
	if (m_selfCalib) m_selfCalib->onFrame(buffer, width, height);
}


// Camera Template functions
// live camera template
int QmlMainWindow::loadTemplatePAS(int camIndex, QString strFilePath)
{ 
	int ret = m_d360Data.parseTemplatePAS(camIndex, strFilePath, m_d360Data.getGlobalAnimSettings().cameraSettingsList());
	if (ret > 0)
		PANO_N_LOG(QString("Loaded template PAS file successfully"));
	else
		PANO_N_LOG(QString("Loading template PAS file failed"));
	return ret;
} 

void QmlMainWindow::saveTemplatePAS(int camIndex, QString strFilePath)
{
	m_d360Data.saveTemplatePASFile(camIndex, strFilePath);
	PANO_N_LOG(QString("Saved template PAS file successfully"));
}

// stitch  camera template
int QmlMainWindow::loadTemplatePAC(QString strFilePath, QList<int> indexList)
{ 
	int ret = m_d360Data.parseTemplatePAC(strFilePath, m_d360Data.getGlobalAnimSettings().cameraSettingsList(), indexList);

	if (indexList.length() > 0) {
		if (ret > 0)
		{
			sharedImageBuffer->getStitcher()->restitch(true);
			PANO_N_LOG(QString("Loaded template PAC file successfully"));
			emit templatePacFileLoadFinished();
		}
		else
			PANO_N_LOG(QString("Loading template PAC file failed"));
	}

	return ret;
} 

void QmlMainWindow::saveTemplatePAC(QString strFilePath, QList<int> indexList)
{ 
	m_d360Data.saveTemplatePACFile(strFilePath, indexList);
	PANO_N_LOG(QString("Saved template PAC file successfully"));
}

int QmlMainWindow::addFavoriteTemplate(QString strFilePath)
{
	QList<QString> favoriteList = loadFavoriteTemplate();
	int ret = saveIniPath(strFilePath);
	if (ret >= 0)
	{
		favoriteList.insert(0, strFilePath);
		m_d360Data.saveTemplateFavoriteList(FAVORITE_XML_FILENAME, favoriteList);
		PANO_N_LOG(QString("Added favorite template successfully"));
		return 0;
	}		
	else
	{
		PANO_N_WARN(QString("Failed to add favorite template"));
		return -1;
	}		
}

void QmlMainWindow::saveLUT(QString strFilePath)
{
	if (QFile(strFilePath).exists())
		QFile(strFilePath).remove();

	
	QSettings settings(strFilePath, QSettings::IniFormat);

	{
		/*    [LUT]
		*     0: Gray, 1 : Red, 2 : Green, 3 : Blue
		*/

		settings.beginGroup("LUT");
		for (int i = 0; i < 4; i++){

			QString str;
			QVariantList *lutData = sharedImageBuffer->getGlobalAnimSettings()->getLutData();

			str.sprintf("%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", lutData[i][0].toFloat(), lutData[i][1].toFloat(), lutData[i][2].toFloat(),
				lutData[i][3].toFloat(), lutData[i][4].toFloat(), lutData[i][5].toFloat(), lutData[i][6].toFloat(),
				lutData[i][7].toFloat(), lutData[i][8].toFloat(), lutData[i][9].toFloat(), lutData[i][10].toFloat());
			switch (i)
			{
			case 0:
				settings.setValue(QLatin1String("Gray"), str);
				break;
			case 1:
				settings.setValue(QLatin1String("Red"), str);
				break;
			case 2:
				settings.setValue(QLatin1String("Green"), str);
				break;
			case 3:
				settings.setValue(QLatin1String("Blue"), str);
				break;
			}
		}
		settings.endGroup();
	}
}

void QmlMainWindow::loadLUT(QString strFilePath)
{
	if (!QFile::exists(strFilePath))
		return;

	QSettings settings(strFilePath, QSettings::IniFormat);

	QVariantList *lutData = sharedImageBuffer->getGlobalAnimSettings()->getLutData();

	{	// [LUT]		
		settings.beginGroup("LUT");
		const QStringList childKeys = settings.childKeys();
		foreach(const QString &childKey, childKeys)
		{
			if (childKey == "Gray")
			{
				QString str = settings.value(childKey).toString();
				QStringList strList = settings.value(childKey).toString().split(",", QString::SkipEmptyParts);
				lutData[0].clear();

				for (int i = 0; i < LUT_COUNT; i++)
				{
					lutData[0].append(strList[i].toFloat());
				}

				grayLutDataChanged(lutData[0]);
				luTChanged(lutData[0], 0);
			}
			else if (childKey == "Red")
			{
				QString str = settings.value(childKey).toString();
				QStringList strList = settings.value(childKey).toString().split(",", QString::SkipEmptyParts);
				lutData[1].clear();

				for (int i = 0; i < LUT_COUNT; i++)
				{
					lutData[1].append(strList[i].toFloat());
				}

				redLutDataChanged(lutData[1]);
				luTChanged(lutData[1], 1);
			}
			else if (childKey == "Green")
			{
				QString str = settings.value(childKey).toString();
				QStringList strList = settings.value(childKey).toString().split(",", QString::SkipEmptyParts);
				lutData[2].clear();

				for (int i = 0; i < LUT_COUNT; i++)
				{
					lutData[2].append(strList[i].toFloat());
				}
				greenLutDataChanged(lutData[2]);
				luTChanged(lutData[2], 2);
			}
			else if (childKey == "Blue")
			{
				QString str = settings.value(childKey).toString();
				QStringList strList = settings.value(childKey).toString().split(",", QString::SkipEmptyParts);
				lutData[3].clear();

				for (int i = 0; i < LUT_COUNT; i++)
				{
					lutData[3].append(strList[i].toFloat());
				}
				blueLutDataChanged(lutData[3]);
				luTChanged(lutData[3], 3);
			}
		}

		settings.endGroup();
	}
}

// parse favorite template list
QList<QString> QmlMainWindow::loadFavoriteTemplate()
{	
	QList<QString> favoriteList = m_d360Data.parseFavoriteTemplateXML(FAVORITE_XML_FILENAME);

	return favoriteList;
}

QList<QString> QmlMainWindow::loadTemplateList()
{
	QList<QString> templateList;

	QDir curPath(QString("./"));
	QDir templatePath(QString("./%1").arg(TEMPLATE_FOLDERNAME));
	if (!templatePath.exists())
		curPath.mkdir(TEMPLATE_FOLDERNAME);
	QStringList fileList = templatePath.entryList(QDir::Files);
	foreach(const QString file, fileList) {
		if (file.endsWith(".l3d"))
		{
			templateList.append(templatePath.absoluteFilePath(file));
		}
			
	}
	return templateList;
}
