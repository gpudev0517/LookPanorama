#pragma once
#include <QObject>
#include "D360Parser.h"
#include "D360Process.h"
#include "MCQmlCameraView.h"
#include "QmlInteractiveView.h"
#include "SharedImageBuffer.h"
#include "CameraView.h"
#include "PlaybackView.h"
#include "D360Process.h"
#include "QmlRecentDialog.h"
#include "OculusViewer.h"
#include <QtXml/QDomDocument>
#include <qvector.h>
#include <vector>
#include "BaseFfmpeg.hpp"
#include "common.h"
#include "CoreEngine.h"
#include "OculusViewer.h"
#include "qfiledialog.h"
#include "CalibProcess.h"
#include "PanoLog.h"
#include "BannerThread.h"
#include "TakeMgrTreeModel.h"
#include <QPointer>
#include "QmlTakeManagement.h"
#include "QmlApplicationSetting.h"
#include "CaptureDevices.h"

extern QString getVideoCodecStringFromType(int videoCodec);
extern int getVideoCodecTypeFromString(QString videoCodecString);
extern QString getAudioCodecStringFromType(int audioCodec);
extern int getAudioCodecTypeFromString(QString audioCodecString);
extern TakeMgrTreeModel* g_takeMgrModel;

class QmlApplicationSetting;
class RecentInfo;
struct SessionTakeInformation;
// 
// enum ViewMode
// {
// 	LIVE_VIEW = 1,
// 	SPHERICAL_VIEW = 2,
// 	STITCH_VIEW = 3
// };
// 
// enum PlayMode
// {
// 	START_MODE = 1,
// 	PAUSE_MODE = 2
// };
// 
// enum BlendMode
// {
// 	FEATHER_MODE = 1,
// 	MULTIBAND_MODE = 2,
// 	WEIGHTVISUALIZER_MODE = 3
// };
// 
// enum OPEN_INI_ERROR
// {
// 	FILE_NO_EXIST = -1,
// 	FILE_VERSION_NO_MATCH = -2
// };
// 
// enum StreamingMode
// {
// 	RTMP = 1,
// 	WEBRTC = 2
// };
// 
// struct MousePos
// {
// 	int x;
// 	int y;
// 	int width;
// 	int height;
// };
// 
// class RecentInfo
// {
// public:
// 	QString		title;
// 	QString		fullPath;
// 	int			type;
// 
// 	bool operator == ( const RecentInfo& other )
// 	{
// 		if( title != other.title )
// 			return false;
// 
// 		if( fullPath != other.fullPath )
// 			return false;
// 
// 		if( type != other.type )
// 			return false;
// 
// 		return true;
// 	}
// 
// 	RecentInfo& operator = ( const RecentInfo& other )
// 	{
// 		title = other.title;
// 		fullPath = other.fullPath;
// 		type = other.type;
// 
// 		return *this;
// 	}
// };
// 
// struct SessionTakeInformation
// {
// 	QString name;
// 	QString filePath;
// 	QString startTime;
// 	QString	duration;
// 	QString comment;
// };

class QmlMainWindow : public QObject
{
	Q_OBJECT
	Q_PROPERTY(QObject *recDialog READ recDialog WRITE setRecDialog NOTIFY recDialogChanged)
	Q_PROPERTY(QObject *takManagement READ takManagement WRITE setTakManagement NOTIFY takManagementChanged)
	Q_PROPERTY(QModelIndex curIndex READ curIndex NOTIFY curIndexChanged)
	Q_PROPERTY(int sliderValue READ sliderValue NOTIFY sliderValueChanged)
	Q_PROPERTY(QString elapsedTime READ elapsedTime NOTIFY elapsedTimeChanged)
	Q_PROPERTY(QString fps READ fps NOTIFY fpsChanged)
	Q_PROPERTY(bool exit READ exit NOTIFY exitChanged)
	Q_PROPERTY(bool start READ start NOTIFY started)
	Q_PROPERTY(QString loadingMessage READ loadingMessage NOTIFY loadingMessageReceived)
	Q_PROPERTY(bool playFinish READ playFinish NOTIFY playFinished)
	Q_PROPERTY(QString errorMsg READ errorMsg NOTIFY error)
	Q_PROPERTY(int xRes READ getXres NOTIFY cameraResolutionChanged)
	Q_PROPERTY(int yRes READ getYres NOTIFY cameraResolutionChanged)
	Q_PROPERTY(bool calibrated READ calibrated NOTIFY calibratingFinished)
	Q_PROPERTY(QString notifyMsg READ notifyMsg NOTIFY notify)
	Q_PROPERTY(bool disconnectComplet READ disconnectComplete NOTIFY disconnectCompleted)
	Q_PROPERTY(bool isEmpty READ disconnectComplete NOTIFY disconnectCompleted)
	Q_PROPERTY(bool templatePacFileLoaded READ templatePacFileLoaded NOTIFY templatePacFileLoadFinished)

	Q_PROPERTY(int weightMapEditCameraIndex READ weightMapEditCameraIndex NOTIFY weightMapEditCameraIndexChanged)
	Q_PROPERTY(bool weightMapEditUndoStatus READ weightMapEditUndoStatus NOTIFY weightMapEditUndoStatusChanged)
	Q_PROPERTY(bool weightMapEditRedoStatus READ weightMapEditRedoStatus NOTIFY weightMapEditRedoStatusChanged)
	Q_PROPERTY(bool weightMapResetStatus READ weightMapResetStatus NOTIFY weightMapResetStatusChanged)
	Q_PROPERTY(QString singleCalibMessage READ singleCalibMessage NOTIFY singleCalibMessageChanged)
	Q_PROPERTY(int singleCalibStatus READ singleCalibStatus NOTIFY singleCalibStatusChanged)
	Q_PROPERTY(float strengthRatio READ strengthRatio NOTIFY strengthRatioChanged)
	Q_PROPERTY(QString strengthDrawColor READ strengthDrawColor NOTIFY strengthDrawColorChanged)
	Q_PROPERTY(QVariantList grayLutData READ grayLutData NOTIFY grayLutDataChanged)
	Q_PROPERTY(QVariantList redLutData READ redLutData NOTIFY redLutDataChanged)
	Q_PROPERTY(QVariantList greenLutData READ greenLutData NOTIFY greenLutDataChanged)
	Q_PROPERTY(QVariantList blueLutData READ blueLutData NOTIFY blueLutDataChanged)
	Q_PROPERTY(QObject *applicationSetting READ applicationSetting WRITE setApplicationSetting NOTIFY applicationSettingChanged)

public:
	QmlMainWindow();
	virtual ~QmlMainWindow();

	QObject *recDialog() const;
	QObject *takManagement() const;
	QObject *applicationSetting() const;
	QObject* singleCalib() const { return m_singleCalib; }
	QModelIndex curIndex() const { return m_curIndex; }
	void setCurIndex(QModelIndex curIndex);
	int sliderValue() const { return m_nSliderValue; }
	void setSliderValue(int value);
	QString elapsedTime() const { return m_eTimeValue; }
	void setETimeValue(QString elapsedTime);
	QString fps() const { return m_fpsValue; }
	void setFpsValue(QString fps);
	bool exit() { return m_isExit; }
	void setExit(bool isExit) { m_isExit = isExit; emit exitChanged(true); }
	bool start() { return m_isStarted; }
	void setStart(bool isStarted) { m_isStarted = isStarted; emit started(true); }
	QString loadingMessage() { return m_loadingMessage; }	
	void setLoadingMessage(QString loadingMessage) { m_loadingMessage = loadingMessage; emit loadingMessageReceived(m_loadingMessage); }	
	bool playFinish() { return m_isPlayFinished; }
	void setPlayFinish(bool isPlayFinished) { m_isPlayFinished = isPlayFinished; emit playFinished(); };
	QString errorMsg() { return m_errorMsg; }
	void setErrorMsg(QString errorMsg) { m_errorMsg = errorMsg; emit error(); }
	bool calibrated() { return m_calibResult; }
	QString notifyMsg() { return m_notifyMsg; }
	bool templatePacFileLoaded() { return m_isTemplatePacFileLoad; }
	void setTemplatePacFileLoaded(bool isLoaded) { m_isTemplatePacFileLoad = isLoaded; }
	QString getPlayBackFilePath() {	return m_strTrackFilePath; }
	QString getCaptureImageFilePath() { return m_strImageFilePath; }
	int weightMapEditCameraIndex() { return m_weightMapEditCameraIndex; }
	void setWeightMapEditCameraIndex(int cameraIndex) { m_weightMapEditCameraIndex = cameraIndex; }
	bool weightMapEditUndoStatus() { return m_weightMapEditUndoStatus; }
	void setWeightMapEditUndoStatus(bool undoStatus) { m_weightMapEditUndoStatus = undoStatus; }
	bool weightMapEditRedoStatus() { return m_weightMapEditRedoStatus; }
	void setWeightMapEditRedoStatus(bool redoStatus) { m_weightMapEditRedoStatus = redoStatus; }
	bool weightMapResetStatus() { return m_weightMapResetStatus; }
	void setWeightMapResetStatus(bool resetStatus) { m_weightMapResetStatus = resetStatus; }
	QString singleCalibMessage() const { return m_singleCalibMessage; }
	int singleCalibStatus() const { return m_singleCalibStatus; }
	float strengthRatio() const { return m_strengthRatio; }
	QString strengthDrawColor() const { return m_strengthDrawColor; }
	QVariantList grayLutData() const { return sharedImageBuffer->getGlobalAnimSettings()->getLutData()[0]; }
	QVariantList redLutData() const { return sharedImageBuffer->getGlobalAnimSettings()->getLutData()[1]; }
	QVariantList greenLutData() const { return sharedImageBuffer->getGlobalAnimSettings()->getLutData()[2]; }
	QVariantList blueLutData() const { return sharedImageBuffer->getGlobalAnimSettings()->getLutData()[3]; }

	QString streamingPath() { return m_d360Data.getGlobalAnimSettings().m_wowzaServer; }

protected:
	D360Parser  m_d360Data;
	D360Process*	m_process;
	
	SharedImageBuffer* sharedImageBuffer;
	
	QMap< int, int > deviceNumberMap;

	QMap<int, AudioThread*> audioModuleMap;
	PlaybackModule* m_playbackModule;

	// for Nodal
	QMap<int, CameraModule*> m_nodalCameraModuleMap;

	QMap< int, CameraModule* > cameraModuleMap;
    QMap<int, DeviceMetaInformation> m_videoDevices;
	QMap<int, QString> m_audioDevices;
		
	//StreamProcess* m_offlineVideoSaveProcess;
	StreamProcess* m_streamProcess;

	std::vector<BannerThread*> m_bannerThreads;
		
	int m_weightMap_cameraIndex;
	int m_weightMap_cameraIndex_;
	int m_weightMap_radius; // 1 ~ 50
	float m_weightMap_strength; // 50 ~ 200
	float m_weightMap_fallOff; // 0.1 ~ 1 (UI: 1 ~ 100)
	bool m_weightMap_isIncrement;
	WeightMapEyeMode m_weightMap_eyeMode;

public:
	MCQmlCameraView* m_playbackView;
	MCQmlCameraView* m_stitcherView;
	MCQmlCameraView* m_calibView;
	MCQmlCameraView* m_stitcherFullScreenView;
	QmlInteractiveView* m_interactView;
	ViewMode		m_viewMode;
	PlayMode m_playMode;
	PlayMode m_playbackMode;
	QModelIndex m_curIndex;
	int m_eyeMode; 
signals:
	void recDialogChanged(const QObject* recDialog);
	void takManagementChanged(const QObject* takeManagement);
	void sliderValueChanged(int sliderValue);
	void elapsedTimeChanged(QString elapsedTime);
	void curIndexChanged(QModelIndex curIndex);
	void fpsChanged(QString fps);
	void levelChanged(QString level);
	void exitChanged(bool isExit);
	void started(bool isStarted);
	void loadingMessageReceived(QString loadingMessage);
	void disconnectCompleted(bool isDisconnectCompleted);
	void error();
	void cameraResolutionChanged();
	void playFinished();
	void calibratingFinished();
	void templatePacFileLoadFinished();
	void notify();

	void weightMapEditCameraIndexChanged(int cameraIndex);
	void weightMapEditUndoStatusChanged(bool isEnable);
	void weightMapEditRedoStatusChanged(bool isEnable);
	void weightMapResetStatusChanged(bool isReset);
	
	void singleCalibChanged(const QObject* singleCalib);
	void stopSingleCaptureChanged(bool isStopedSingleCapture);
	void singleCalibMessageChanged(const QString messageText);
	void singleCalibStatusChanged(const int status);
	void strengthRatioChanged(const float strengthRatio);
	void strengthDrawColorChanged(const QString color);
	void grayLutDataChanged(const QVariantList lutData);
	void redLutDataChanged(const QVariantList lutData);
	void greenLutDataChanged(const QVariantList lutData);
	void blueLutDataChanged(const QVariantList lutData);
	void applicationSettingChanged(const QObject* applicationSetting);
	
protected:
	int getDeviceIndex(QString name, int type = 0);		// type:0(video), type:1(audio)

	void Stop();
	void Pause();
	void Start();

	void releaseThreads();

public:
	bool attach(CameraInput& camSettings, int deviceNumber, float startframe, float endframe, float fps);
	bool attachNodalCamera(CameraInput& camSettings, int deviceNumber,  int nCameras);
	void disconnectCamera(int index);
	void disconnectCameras();
	GlobalAnimSettings& getGlobalAnimSetting(){ return m_d360Data.getGlobalAnimSettings(); }

	void sendWeightMapEditCameraIndexChanged(int cameraIndex) { m_weightMapEditCameraIndex = cameraIndex; emit weightMapEditCameraIndexChanged(cameraIndex); }
	void sendWeightMapEditUndoStatusChanged(bool isEnable) { m_weightMapEditUndoStatus = isEnable; emit weightMapEditUndoStatusChanged(isEnable); }
	void sendWeightMapEditRedoStatusChanged(bool isEnable) { m_weightMapEditRedoStatus = isEnable; emit weightMapEditRedoStatusChanged(isEnable); }
	void sendWeightMapResetStatusChanged(bool isReset) { m_weightMapResetStatus = isReset; emit weightMapResetStatusChanged(isReset); }
	void sendSingleCalibMessageChanged(QString messageText) { m_singleCalibMessage = messageText; emit singleCalibMessageChanged(messageText); }
	void sendSingleCalibStatusChanged(int singleCalibStatus) { m_singleCalibStatus = singleCalibStatus; emit singleCalibStatusChanged(singleCalibStatus); }
	void sendStrengthRatioChanged(float strengthRatio) { m_strengthRatio = strengthRatio; emit strengthRatioChanged(strengthRatio); }
	void sendStrengthDrawColorChanged(QString color) { m_strengthDrawColor = color; emit strengthDrawColorChanged(color); }

	void newCalibFrame(unsigned char* buffer, int width, int height);

private:
	QString m_Name;
	bool m_isExit;
	bool m_isClose;
	bool m_isStarted;
	bool m_isStopPlayback;

	QString m_loadingMessage;
	bool m_isDisconnectComplete;
	bool m_isPlayFinished;
	bool m_isAboutToStop;
	bool m_isEndCaptureThreads;
	bool m_calibResult;
	QString m_errorMsg;
	QPointer<QmlRecentDialog> m_qmlRecentDialog;
	int m_cameraCnt;
	int m_audioCnt;
	bool m_bblendview;
	bool m_connectedCameras;
	bool m_reconnectCameras;

	int m_weightMapEditCameraIndex;
	bool m_weightMapEditUndoStatus;
	bool m_weightMapEditRedoStatus;
	bool m_weightMapResetStatus;
	QString m_singleCalibMessage;
	int m_singleCalibStatus;
	float m_strengthRatio;
	QString m_strengthDrawColor;

	QMap<int, QString> m_videoPathList;
	QMap<int, QString> m_imagePathList;
	QMap<int, QString> m_cameraNameList;
	QMap<int, QString> m_audioNameList;
	QList<RecentInfo> m_recentList;
	RecentInfo m_recent;
	QLocale		conv;
	int			m_nSliderValue;
	QString		m_eTimeValue;
	QString		m_fpsValue;
	QString		m_yawValue;
	QString		m_pitchValue;
	QString		m_rollValue;
	OculusRender* m_oculusDevice;
	QStringList m_iPathList;
	CalibProcess* m_calib;
	SingleCameraCalibProcess* m_selfCalib;

	int			m_deletableCameraCnt;
	int			m_deletableAudioCnt;
	QList<CPOINT> m_cpList;
	QList<QString>	m_notifyList;

	PanoLog		m_logger;
	QString		m_notifyMsg;
	bool		m_isTemplatePacFileLoad;

	// Take Management
	QString		m_sessionRootPath;	
	SessionTakeInformation	m_sessionTake;
	int			m_lastFramesProcessed;
	bool		m_isNewConfiguration;
	std::vector<QString> m_sessionNameList;
	std::vector<QString> m_takeNameList;
	QList<QStringList> m_CommentList;
	QPointer<QmlTakeManagement> m_qmlTakeManagement;
	QPointer<QmlApplicationSetting> m_qmlApplicationSetting;
	QPointer<SingleCameraCalibProcess> m_singleCalib;
	//SingleCameraCalibProcess* m_singleCalib;

	QString m_templateIniFile;
	QList<QMap<QString, int>> m_templateOrderList;

	int       m_cameraIndex;
	
	// Playback Management
	QString m_strTrackFilePath;
	QString m_strImageFilePath;
	int		m_nDurationFrame;

public slots:
	bool connectPlayback();
	void disconnectPlayback();
	bool disconnectComplete() { return m_isDisconnectComplete; }
	void setDisconnectComplete(bool disconnectComplete) { m_isDisconnectComplete = disconnectComplete; m_connectedCameras = false; emit disconnectCompleted(m_isDisconnectComplete); }
	void connectToCameras();
	void finishedThread(int type, QString msg = "", int id = -1);
	void startedThread(int type, QString msg = "", int id = -1);
	void reportError(int type, QString msg = "", int id = -1);
	void setRecDialog(QObject *recDialog);
	void setTakManagement(QObject *takeManagement);
	int  openIniPath(QString iniPath);
	void saveWeigthMap(QString iniPath);
	int  saveIniPath(QString iniPath);
	bool openRecentMgrToINI();
	void saveRecentMgrToINI();
	void closeMainWindow(); 
	void receiveKeyIndex(int level);
	int  level();
	int  reloadCameraCalibrationFile(QString strCalibFile);
	bool calibrate(int step = -1);
	void startCalibrating();
	void finishedSnapshot(int id);
	void finishedSphericalSnapshot();
	void startSingleCameraCalibration();
	void finishSingleCameraCalibration();
	void startSingleCapture();
	void stopSingleCapture();
	void startSingleCalibrating();
	bool singleCalibrate();
	void applySingle();
	void setSingleParams(int camIndex, int lensTypeIndex,
		int boardSizeW, int boardSizeH, int snapshotNumber);
	//Recent Title, Path
	QString getRecentTitle(int index){ return m_recentList[index].title; }
	QString getRecentPath(int index);
	QString getRecentFullPath(int index){ return m_recentList[index].fullPath; }
	int getRecentType(int index) { return m_recentList[index].type; }
	// return value: -2 if ini file open failed, -1 if this file is not registered to recent list yet
	int  recentOpenIniPath(QString sourcepath);
	void deleteRecentList(QString title);

	void clearTemplateSetting();
	int  openTemplateIniFile(QString iniFile);
	void setTemplateOrder(QList<QString> strSlotList, QList<int> orderList);
	QList<QString> getTemplateOrderMapList();
	void clearTemplateOrderList() { m_templateOrderList.clear(); }
	int getRigTemplateCameraCount() { return m_d360Data.getRigTemplateGlobalAnimSettings().m_cameraCount; }

	void updateCameraView(QObject* camView, int deviceNum);
	void disableCameraView(QObject* camView);
	void enableCameraView(QObject* camView, bool enabled);
	void updatePlaybackView(QObject* stitchView);
	void updateStitchView(QObject* stitchView);
	void updateCalibCameraView(QObject* calibView);
	void updateFullScreenStitchView(QObject* stitchView, int screenNo);
	void setFullScreenStitchViewMode(int screenNo);
	void updateInteractView(QObject* interactView);
	void updatePreView(QObject* preView);
	int  getRecentCnt() { return m_recentList.size(); }
	QString getMicDeviceName(int index);
	QString getCameraDeviceName(int index);
	QString getCameraDevicePath(int index);
	QString getCameraName(int index) { return m_cameraNameList[index]; }
	int getVideoCameraIndex() { return m_cameraIndex; }
	QString getCameraDeviceNameByPath(QString devicePath);
	QString getDeviceNameByDevicePath(QString devicePath, int& dupIndex);
	int getCameraCnt() { return m_cameraCnt; }
	void setCameraCnt(int numCameraCnt) { m_cameraCnt = numCameraCnt; }
	int getAudioCnt() { return m_audioCnt;}
	int getAudioSelCnt();
	int checkAudioExist(QString videoFilePath);
	void openProject();
	void disconnectProject();
	void setCurrentMode(int);
	void setPlayMode(int);
	void setPlaybackMode(int playMode);
	void stopPlayback();
	void setSeekValue(int nValue);
	void setPlaybackBackward();
	void setPlaybackForward();
	void setPlaybackPrev();
	void setPlaybackNext();

	void setBlendMode(int);
	int  getBlendMode();
	void startWithLiveGrabber(int selectedCameraIndex);
	void takeSnapshot();
	void luTChanged(const QVariantList &map, int colorType);
	QString getTemplateIniFileName() { return m_templateIniFile; }

	// Pause the stream individually for take management function
	void pauseStreamProcess() {		
		if (sharedImageBuffer->getStreamer())
			sharedImageBuffer->getStreamer()->playAndPause(true);
	}

	//Template
	void sendVideoPath(int slotIndex, QString videoPath);
	void setForegroundSlot(int slotIndex);
	QString getVideoPath(int slotIndex) { 

		QMap<int, QString> videoFilePathMap = m_d360Data.getGlobalAnimSettings().m_videoFilePathMap;
		if (videoFilePathMap.contains(slotIndex))  {
			return videoFilePathMap[slotIndex];
		}

		return "";
	}
	int getVideoCount() { 
		return m_videoPathList.size(); 
	}

	// need to show all live cameras on UI
	void setLiveCamera(int slotIndex, QString strDevicePath);
	void setLiveStereo(int slotIndex, int stereoType);
	QString getLiveCamera(int slotIndex);
	int getLiveStereo(int slotIndex);
	int getLiveCameraCount();
	void setTotalIndexAndSelectedIndex(int totalIndex, int selectedIndex);
	int getSelectedIndex(int totalIndex);

	void sendCameraName(int slotIndex, QString cameraName, bool isNodalInput, bool isVideoFile);
	bool isSelectedCamera(QString name);
	void sendAudioName(int slotIndex, QString audioName);
	bool isSelectedAudio(QString name);
	void sendImagePath(int slotIndex, QString imagePath);
	QString getImagePath(int slotIndex) {
		if (m_imagePathList.contains(slotIndex))
			return m_imagePathList[slotIndex];
		return "";
	}
	int getImageCount() { return m_imagePathList.size(); }
	void initTemplateImageObject(){ if (m_imagePathList.size() > 0){ m_imagePathList.clear(); } }
	void initTemplateVideoObject(){ if (m_videoPathList.size() > 0){ m_videoPathList.clear(); } }
	void initTemplateCameraObject(){ if (m_cameraNameList.size() > 0){ m_cameraNameList.clear(); m_audioNameList.clear(); } }
	int  getTemplateImageCount(){ return m_imagePathList.size(); }
	int  getTemplateVideoCount(){ return m_videoPathList.size(); }
	int  getTemplateCameraCount(){ return m_cameraNameList.size(); }
	void openTemplateVideoIniFile(QList<QString> strSlotList, QList<int> orderList);
	void openTemplateCameraIniFile(QList<QString> strSlotList, QList<int> orderList);
	void openTemplateImageIniFile(QList<QString> strSlotList, QList<int> orderList);

	void streamPanorama(unsigned char* panorama);
	void streamAudio(int devNum, void* audioFrame);

	void streamClose();

	void updateStitchingThreadStats(struct ThreadStatisticsData statData);
	void updatePlaybackStitchingThreadStats(struct ThreadStatisticsData statData);

	//startFrame, endFrame
	float startFrame()
	{
		return m_d360Data.startFrame();
	}
	void setStartFrame(float fStartFrame) { m_d360Data.setStartFrame(fStartFrame);}

	float endFrame()
	{
		return m_d360Data.endFrame();
	}

	void setEndFrame(float fEndFrame) { m_d360Data.setEndFrame(fEndFrame);}

	//version
	float fileVersion()
	{
		return m_d360Data.fileVersion();
	}
	//captureType
	int getCaptureType() {
		return m_d360Data.getCaptureType();
	}
	void setCaptureType(int captureType){
		m_d360Data.setCaptureType(captureType);
	}

	//cameraCount
	int getCameraCount() { return m_d360Data.getCameraCount(); }
	void setCameraCount(int cameraCount) { m_d360Data.setCameraCount(cameraCount); }

	//fps
	float getFps() { return m_d360Data.getFps(); }
	void setFps(float fps) { m_d360Data.setFps(fps); }

	//source fps
	float getSourceFps() { return m_d360Data.getSourceFps(); }
	void setSourceFps(float sourcefps) { m_d360Data.setSourceFps(sourcefps); }

	//input resolution
	int getXres() { return m_d360Data.getXres(); }
	void setXres(int xres) { m_d360Data.setXres(xres); }
	int getYres() { return m_d360Data.getYres(); }
	void setYres(int yres) { m_d360Data.setYres(yres); }	

	//cameraClib
	QString getCameraCalib() {
		return m_d360Data.getCameraCalib(); 
	}
	void setCameraCalib(QString cameraCalib) { m_d360Data.setCameraCalib(cameraCalib); }

	//sample Fmt
	int getSampleFmt() { return m_d360Data.getSampleFmt(); }
	void setSampleFmt(int sampleFmt) { m_d360Data.setSampleFmt(sampleFmt); }

	//lag
	int getLag() { return m_d360Data.getLag(); }
	void setLag(int lag) { m_d360Data.setLag(lag); }

	//output resolution
	int getPanoXres() { return m_d360Data.getPanoXres(); }
	void setPanoXres(int panoXres) { m_d360Data.setPanoXres(panoXres); }
	int getPanoYres() { return m_d360Data.getPanoYres(); }
	void setPanoYres(int panoYres) { m_d360Data.setPanoYres(panoYres); }

	//device Count
	int getDeviceCount() { return m_d360Data.getGlobalAnimSettings().m_cameraCount; }
	void setDeviceCount(int deviceCount) { }	
	
	void setBlendLevel(int iBlendLevel);
	int getBlendLevel();

	// Crop 
	float  getLeft(int iDeviceNum);
	void setLeft(float fLeft, int iDeviceNum);
	float getRight(int iDeviceNum);
	void setRight(float fRight, int iDeviceNum);
	float getTop(int iDeviceNum);
	void setTop(float fTop, int iDeviceNum);
	float getBottom(int iDevicNum);
	void setBottom(float fBottom, int iDeviceNum);

	//stereo
	int getStereoType(int);
	void setStereoType(int iStereoType, int iDeviceNum);
	bool isLeftEye(int iDeviceNum);
	bool isRightEye(int iDeviceNum);
	//audioType
	void setAudioType(int iAudioType, int iDeviceNum);
	//exposure
	float getCameraExposure(int iDeviceNum) { 
		return m_d360Data.getGlobalAnimSettings().getCameraInput(iDeviceNum).exposure;
	}
	void setCameraExposure(int iDeviceNum, float fExposure) { 
		m_d360Data.getGlobalAnimSettings().getCameraInput(iDeviceNum).exposure = fExposure;
		reStitch();
	}
	void onCalculatorGain();
	void onRollbackGain();
	void onResetGain();
	void setTempCameraSettings() { m_d360Data.setTempCameraSettings(); }
	void onCancelCameraSettings() { 
		m_d360Data.resetCameraSettings(); 
		reStitch(true);
	}

	//Calibration
	void setCalibFile(QString calibFile) { m_d360Data.getGlobalAnimSettings().setCameraCalib(calibFile); }
	QString getCalibFile()  { return m_d360Data.getGlobalAnimSettings().m_cameraCalibFile; }
	int getLensType() { return m_d360Data.getGlobalAnimSettings().m_lensType; }
	void setLensType(int lensType) { m_d360Data.getGlobalAnimSettings().m_lensType = (CameraParameters::LensType)lensType; }
	int getFov() { return m_d360Data.getGlobalAnimSettings().m_fov; }
	void setFov(int fov) { m_d360Data.getGlobalAnimSettings().m_fov = fov; }
	
	//oculus
	void enableOculus(bool);
	bool getOculus() { return m_d360Data.getGlobalAnimSettings().m_oculus && m_oculusDevice != NULL && m_oculusDevice->isConnected(); }
	void showChessboard(bool);

	//Slots for template settings
	void setTempStereoType(int iIndex, int iStereoType);
	int getTempStereoType(int iIndex);
	void setTempImagePath(int iIndex, QString fileDir);
	QString getTempImagePath(int iIndex);
	void setTempImagePrefix(int iIndex, QString filePrefix);
	QString getTempImagePrefix(int iIndex);
	void setTempImageExt(int iIndex, QString fileExt);
	QString getTempImageExt(int iIndex);
	void setTempAudioSettings(int iIndex, int iAudioType);
	int getTempAudioSettings(int iIndex);
	int getTempAudioSettingsEx(QString devName);
	int getTempWidth() { return m_d360Data.getTempGlobalAnimSettings().m_xres;}
	void setTempWidth(int width) { m_d360Data.getTempGlobalAnimSettings().setXres(width); }
	int getTempHeight() { return m_d360Data.getTempGlobalAnimSettings().m_yres; }
	void setTempHeight(int height) { m_d360Data.getTempGlobalAnimSettings().setYres(height); }
	int getTempPanoWidth() { return m_d360Data.getTempGlobalAnimSettings().m_panoXRes; }
	void setTempPanoWidth(int width) { m_d360Data.getTempGlobalAnimSettings().setPanoXres(width); }
	int getTempPanoHeight()  { return m_d360Data.getTempGlobalAnimSettings().m_panoYRes; }
	void setTempPanoHeight(int height) { m_d360Data.getTempGlobalAnimSettings().setPanoYres(height); }
	float getTempFPS()  { return m_d360Data.getTempGlobalAnimSettings().m_fps; }
	void setTempFPS(float fps) { m_d360Data.getTempGlobalAnimSettings().setFps(fps); }
	float getTempSourceFPS()  { return m_d360Data.getTempGlobalAnimSettings().m_sourcefps; }
	void setTempSourceFPS(float sourceFps) { m_d360Data.getTempGlobalAnimSettings().setSourceFps(sourceFps); }
	QString getTempCalibFile()  { return m_d360Data.getTempGlobalAnimSettings().m_cameraCalibFile; }
	void setTempCalibFile(QString calibFile) { m_d360Data.getTempGlobalAnimSettings().setCameraCalib(calibFile); }
	int getTempStartFrame()  { return m_d360Data.getTempGlobalAnimSettings().m_startFrame; }
	void setTempStartFrame(int nFrame) { m_d360Data.getTempGlobalAnimSettings().setStartFrame(nFrame); }
	int getTempEndFrame()  { return m_d360Data.getTempGlobalAnimSettings().m_endFrame; }
	void setTempEndFrame(int nFrame) { m_d360Data.getTempGlobalAnimSettings().setEndFrame(nFrame); }
	QString getTempFileExt() { return m_d360Data.getTempGlobalAnimSettings().m_fileExt; }
	void setTempFileExt(QString fileExt) {	m_d360Data.getTempGlobalAnimSettings().setFileExt(fileExt); }
	bool getTempOculus() { return m_d360Data.getTempGlobalAnimSettings().m_oculus; }
	void setTempOculus(bool oculus) { m_d360Data.getTempGlobalAnimSettings().m_oculus = oculus; }
	QString getTempOfflineVideoCodec() { return getVideoCodecStringFromType(m_d360Data.getTempGlobalAnimSettings().m_videoCodec); }
	void setTempOfflineVideoCodec(QString codec) { m_d360Data.getTempGlobalAnimSettings().m_videoCodec = getVideoCodecTypeFromString(codec); }
	QString getTempOfflineAudioCodec() { return getAudioCodecStringFromType(m_d360Data.getTempGlobalAnimSettings().m_audioCodec); }
	void setTempOfflineAudioCodec(QString codec) { m_d360Data.getTempGlobalAnimSettings().m_audioCodec = getAudioCodecTypeFromString(codec); }
	QString getTempStreamVideoCodec() { return getVideoCodecStringFromType(m_d360Data.getTempGlobalAnimSettings().m_videoCodec); }
	void setTempStreamVideoCodec(QString codec) { m_d360Data.getTempGlobalAnimSettings().m_videoCodec = getVideoCodecTypeFromString(codec); }
	QString getTempStreamAudioCodec() { return getAudioCodecStringFromType(m_d360Data.getTempGlobalAnimSettings().m_audioCodec); }
	void setTempStreamAudioCodec(QString codec) { m_d360Data.getTempGlobalAnimSettings().m_audioCodec = getAudioCodecTypeFromString(codec); }
	void setLidarPort( float lidatPort ){m_d360Data.getTempGlobalAnimSettings().m_lidarPort = lidatPort;}
	float getLidarPort(){return m_d360Data.getTempGlobalAnimSettings().m_lidarPort;}
	// Auto Calibrating...
	int getTempLensType() { return m_d360Data.getTempGlobalAnimSettings().m_lensType; }
	void setTempLensType(int lensType) { m_d360Data.getTempGlobalAnimSettings().m_lensType = (CameraParameters::LensType)lensType; }
	int getTempFov() { return m_d360Data.getTempGlobalAnimSettings().m_fov; }
	void setTempFov(int fov) { m_d360Data.getTempGlobalAnimSettings().m_fov = fov; }

	void resetTempGlobalSettings() { m_d360Data.resetTempGlobalAnimSettings(); }
	void clearTempGlobalSettings() { m_d360Data.clearTempGlobalAnimSettings(); }

	//snapshot
	QString getSnapshotDir() { 
		return m_d360Data.getGlobalAnimSettings().m_snapshotDir; 
	}
	void setSnapshotDir(QString snapshotDir) { 
		m_d360Data.getGlobalAnimSettings().m_snapshotDir = snapshotDir;
		sharedImageBuffer->getGlobalAnimSettings()->m_snapshotDir = snapshotDir;
	}

	QString getTempSnapshotDir() {
		return m_d360Data.getTempGlobalAnimSettings().m_snapshotDir;
	}
	void setTempSnapshotDir(QString snapshotDir) {
		m_d360Data.getTempGlobalAnimSettings().m_snapshotDir = snapshotDir;
	}

	//WeightMap
	QString getWeightMapDir() {
		return m_d360Data.getGlobalAnimSettings().m_weightMapDir;
	}
	void setWeightMapDir(QString weightMapDir) {
		m_d360Data.getGlobalAnimSettings().m_weightMapDir = weightMapDir;
		sharedImageBuffer->getGlobalAnimSettings()->m_weightMapDir = weightMapDir;
	}

	QString getTempWeightMapDir() {
		return m_d360Data.getTempGlobalAnimSettings().m_weightMapDir;
	}
	void setTempWeightMapDir(QString weightMapDir) {
		m_d360Data.getTempGlobalAnimSettings().m_weightMapDir = weightMapDir;
	}

	void snapshotFrame();
	void snapshotPanoramaFrame();
	void reStitch(bool cameraParamChanged = false) { sharedImageBuffer->getStitcher()->restitch(cameraParamChanged); }	
	void onMovedSpherical(int, int);
	void onReleasedSpherical(int, int);
	void onPressedSpherical(int, int);
	void onDoubleClickedSpherical(int, int);
	void onMovedInteractive(int, int);
	void onReleasedInteractive(int, int);
	void onPressedInteractive(int, int);
	QString onImageFileDlg(QString);
	int getCaptyreType() { return m_d360Data.getGlobalAnimSettings().m_captureType; }
	void resetConfigList();

	// Control points...
	int getCPointCount(int camIndex1, int camIndex2);
	QString getCPoint(int index, int camIndex1, int camIndex2);	// Return format -> "x1:y1:x2:y2"
	int getCPointCountEx(int camIndex);
	QString getCPointEx(int index, int camIndex);	// Return format -> "x:y:index"
	void initCPointList(){ if (m_cpList.size() > 0){ m_cpList.clear(); } }

	// Seam view...
	QList<qreal> getSeamLabelPos(int camIndex);
	bool enableSeam(int camIndex1 = -1, int camIndex2 = -10);		// -1: all disbable, 0: view all camera seams  
																	//  if Mirror, camIndex1 will be LeftCamera, camera2 will be rightCamera
																	//  if not Mirror, camIndex will be -10.

	// Notification
	void onNotify(int type, QString title, QString msg);
	int getNotificationCount();
	QString getNotification(int index);
	bool removeNotification(int index);		// -1: all remove

	/// <summary>
	/// Get slot info
	/// </summary>
	/// <param name="type">
	/// CAPTURE_DSHOW = 1
	/// CAPTURE_VIDEO = 2
	/// CAPTURE_FILE = 3
	/// </param>
	/// <returns>
	/// List of input width, height, and frame rate of slot.
	/// Empty list if slot is unavailable
	/// </returns>
	QList<qreal> getSlotInfo(QString name, QString ext, int type);	// name: video or image file name or DirectShow camera name

	// Save file quality
	int getQuality() { return m_d360Data.getTempGlobalAnimSettings().m_crf; }
	void setQuality(int quality) { m_d360Data.getTempGlobalAnimSettings().m_crf = quality; }

	// Use NVidia
	void enableNvidia(bool isGpu) { m_d360Data.getGlobalAnimSettings().m_isGPU = isGpu; }
	bool isNvidia() { return m_d360Data.getGlobalAnimSettings().m_isGPU; }

	// Banner
	bool addBanner(int windowWidth, int windowHeight, QPoint pt0, QPoint pt1, QPoint pt2, QPoint pt3, QString bannerFile, bool isVideo);
	bool addBanner(vec2 quad[], QString bannerFile, bool isVideo, bool isStereoRight);
	void removeAllBanners();
	void removeLastBanner();
	void removeBannerAtIndex(int index);
	QString getPosWindow2Pano(int w,int h,int x,int y);
	QString getPosPano2Window(int w, int h, int x, int y);
	int getBannerCount() { return m_d360Data.getGlobalAnimSettings().m_banners.size(); }

	void resetCamSettings();
	bool isStereo() { return m_d360Data.getGlobalAnimSettings().isStereo(); }

	// Screen information
	int getScreenCount();	
	QString getFullScreenInfoStr(int screenNo);
	QRect getFullScreenInfo(int screenNo);

	// WeightMap functions
	QString getWeightMapFile(int cameraIndex);
	void setUpdatedWeightMapFile(QString strFilename);

	void setWeightMapEditMode(bool isEditMode);
	void setDrawWeightMapSetting(int cameraIndex, int cameraIndex_,  int radius, float strength, float fallOff, bool isIncrement, int nEyeMode);
	void drawWeightMap(int w, int h, int x, int y);	
	void setWeightMapChanged(bool isChanged);
	void saveWeightMapToFile(int *weightMap, int width, int height);
	void resetWeightMap();	
	void weightmapUndo();
	void weightmapRedo();
	void setWeightMapPaintingMode(int paintMode);

	// close project
	void setCloseProject(bool isClose) { m_isClose = isClose; };
	bool isCloseProject() { return m_isClose; };

	// CT functions
	void setColorTemperature(int ctValue);
	int getColorTemperature() { return (int)sharedImageBuffer->getStitcher()->getColorTemperature(); }

	// create/delete session/take
	void setSessionRootPath(QString dirPath);
	QString getSessionRootPath();
	int getLastSessionId();
	int getLastTakeId(int sessionId);
	void createNewSession();
	bool isTakeNode(QModelIndex index);
	QString getTakeComment(QModelIndex index);
	void changeComment(QModelIndex index, QString strComment);
	bool startRecordTake();
	bool stopRecordTake(QString strComment);
	void deleteSession(QString strSessionPath);
	void deleteTake(QString strTakePath);
	QString travelSessionRootPath();
	void removeAllSessions(TakeMgrTreeModel* model,int position, int sessionCount);

	// RTMP Streaming
	bool startStreaming(QString rtmpAddress, int streamingWidth, int streamingHeight, int streamingMode);
	bool stopStreaming();
	void setStreamingPath(QString rtmpAddress);
	QString getStreamingPath();

	bool isRecordAvailable() {
		if (m_streamProcess)
		{
			PANO_N_LOG("Please stop streaming function first.");
			return false;
		}			
		return true;
	}

	// Playback Management
	void initPlayback(QModelIndex index);
	void resetPlayback(QModelIndex index);

	void setDurationString(int nFrames);
	QString getDurationString();
	int getDuration();

	// Nodal Shooting	
	void setNodalVideoFilePath(int slotIndex, QString strVideoFilePath);	
	void setNodalMaskImageFilePath(int soltIndex, QString strMaskImageFilePath);
	QString getNodalMaskImageFilePath(int slotIndex);
	QString getNodalVideoFilePath(int slotIndex);
	void setNodalCameraIndex(int index);
	int  getNodalVideoIndex() { 
		return m_d360Data.getGlobalAnimSettings().m_nodalVideoIndex; 
	}
	int getNodalVideoCount() {
		return m_d360Data.getGlobalAnimSettings().m_nodalVideoFilePathMap.values().size();
	}
	void setNodalSlotIndex(int nodalSlotIndex) {
		m_d360Data.getGlobalAnimSettings().m_nodalSlotIndexMap[nodalSlotIndex] = nodalSlotIndex;
	}

	int getNodalSlotIndex(int slotIndex) {
		if (!m_d360Data.getGlobalAnimSettings().m_nodalSlotIndexMap.contains(slotIndex))
			return -4;

		return m_d360Data.getGlobalAnimSettings().m_nodalSlotIndexMap[slotIndex];
	}

	bool isNodalConfiguration() { return m_d360Data.getGlobalAnimSettings().m_nodalVideoIndex >= 0; }

	// Camera Parameters
	float  getYaw(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_yaw; }
	void setYaw(float yaw, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_yaw = yaw; }
	float getPitch(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_pitch; }
	void setPitch(float pitch, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_pitch = pitch; }
	float getRoll(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_roll; }
	void setRoll(float roll, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_roll = roll; }
	int getLensType(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_lensType; }
	float getFov(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_fov; }
	void setFov(float fov, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_fov = fov; }
	float getFovy(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_fovy; }
	void setFovy(float fovy, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_fovy = fovy; }
	float getK1(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_k1; }
	void setK1(float k1, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_k1 = k1; }
	float getK2(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_k2; }
	void setK2(float k2, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_k2 = k2; }
	float getK3(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_k3; }
	void setK3(float k3, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_k3 = k3; }
	float getOffsetX(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_offset_x; }
	void setOffsetX(float offset_x, int deviceNum) {m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_offset_x = offset_x; }
	float getOffsetY(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_offset_y; }
	void setOffsetY(float offset_y, int deviceNum) {m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_offset_y = offset_y;}
	float getExpOffset(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_expOffset; }
	void setExpOffset(float expOffset, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).m_cameraParams.m_expOffset = expOffset; }
	int getWidth(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).xres; }
	int getHeight(int deviceNum) { return m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).yres; }
	void setWidth(int width, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).xres = width; }
	void setHeight(int height, int deviceNum) { m_d360Data.getGlobalAnimSettings().getCameraInput(deviceNum).yres = height; }

	QString getTempSplitMins() { return QString::number(m_d360Data.getTempGlobalAnimSettings().m_splitMins); }
	void setTempSplitMins(QString splitMins) { qDebug() << splitMins; m_d360Data.getTempGlobalAnimSettings().m_splitMins = splitMins.toFloat(); }

	// live  camera template
	int loadTemplatePAS(int camIndex, QString strFilePath); 
	void saveTemplatePAS(int camIndex, QString strFilePath); 

	// stitch  camera template
	int loadTemplatePAC(QString strFilePath, QList<int> indexList);	
	void saveTemplatePAC(QString strFilePath, QList<int> indexList);

	// Favorite template
	int addFavoriteTemplate(QString strFilePath);
	QList<QString> loadFavoriteTemplate();
	QList<QString> loadTemplateList();
	// Application Setting
	void setApplicationSetting(QObject* applicationSetting);
	QString getDisplayName(QString l3dPath);
	
	//LUT
	void saveLUT(QString strFilePath);
	void loadLUT(QString strFilePath);

	// init  Devices
	void initDevices();


private:
	bool openIniFileAndUpdateUI(QString projFile);
};