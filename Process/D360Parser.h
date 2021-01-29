#pragma once

#include <QObject>
#include <QDebug>
#include <vector>
#include <QMap>
#include <QSettings>
#include <QtXml/QDomDocument>

#include "libavcodec/avcodec.h"
#include "Structures.h"
#include "define.h"

class TemplateModes : public QObject
{
	Q_OBJECT

public:
	enum class Mode {
		LIVE,
		VIDEO,
		IMAGE
	};
	Q_ENUMS(Mode);
	
	static void init();
};

class GlobalAnimSettings : public QObject
{
	Q_OBJECT
	Q_PROPERTY(int fps READ getFps WRITE setFps NOTIFY stateChanged)
	Q_PROPERTY(int xres READ getXres WRITE setXres NOTIFY stateChanged)
	Q_PROPERTY(int yres READ getYres WRITE setYres NOTIFY stateChanged)
	Q_PROPERTY(QString cameraCalib READ getCameraCalib WRITE setCameraCalib NOTIFY stateChanged)
	Q_PROPERTY(int sampleFmt READ getSampleFmt WRITE setSampleFmt NOTIFY stateChanged)
	Q_PROPERTY(int lag READ getLag WRITE setLag NOTIFY stateChanged)
	Q_PROPERTY(int panoXres READ getPanoXres WRITE setPanoXres NOTIFY stateChanged)
	Q_PROPERTY(int panoYres READ getPanoYres WRITE setPanoYres NOTIFY stateChanged)
	Q_PROPERTY(int deviceCount READ getDeviceCount WRITE setDeviceCount NOTIFY stateChanged)

signals:
	void stateChanged();
public slots:
	void set(QObject* settings) {
		operator = (*(qobject_cast<GlobalAnimSettings*>(settings)));
	}
	QObject* get() { return (QObject*)this; }
	//cameraCount
	int getCameraCount() { return m_cameraCount; }
	void setCameraCount(int cameraCount) { m_cameraCount = cameraCount; }
	//fps
	float getFps() { return m_fps; }
	void setFps(float fps) { m_fps = fps; }

	//fps
	float getRealFps() { return m_realFps; }
	void setRealFps(float fps) { m_realFps = fps; }
	
	// source fps
	float getSourceFps() { return m_sourcefps; }
	void setSourceFps(float sourcefps) { m_sourcefps = sourcefps; }

	//input resolution
	int getXres() { return m_xres; }
	void setXres(int xres) { m_xres = xres; }
	int getYres() { return m_yres; }
	void setYres(int yres) { m_yres = yres; }
	//cameraClib
	QString getCameraCalib() { return m_cameraCalibFile; }
	void setCameraCalib(QString cameraCalib) { m_cameraCalibFile = cameraCalib; }
	//sample Fmt
	int getSampleFmt() { return m_sampleFmt; }
	void setSampleFmt(int sampleFmt) { m_sampleFmt = sampleFmt; }
	//lag
	int getLag() { return m_audioLag; }
	void setLag(int lag) { m_audioLag = lag; }
	//output resolution
	int getPanoXres() { return m_panoXRes; }
	void setPanoXres(int panoXres) { m_panoXRes = panoXres; }
	int getPanoYres() { return m_panoYRes; }
	void setPanoYres(int panoYres) { m_panoYRes = panoYres; }
	//device Count
	int getDeviceCount() { return m_cameraCount; }
	void setDeviceCount(int deviceCount) { }

	// Start/End
	int getStartFrame() { return m_startFrame; }
	void setStartFrame(int nFrame) { m_startFrame = nFrame; }
	int getEndFrame() { return m_endFrame; }
	void setEndFrame(int nFrame) { m_endFrame = nFrame; }
    // File Ext
	QString getFileExt() { return m_fileExt; }
	void setFileExt(QString fileExt) { m_fileExt = fileExt; }

	bool isNodalAvailable();
	bool isLiveNodal();

	CameraInput& getCameraInput(int deviceNumber);


	//lidar Port   ???
	int getLiDARPort()
	{
		return m_lidarPort;
	}
	void setLiDARPort( int port )
	{
		m_lidarPort = port;
	}

public:

	typedef std::vector<CameraInput> CameraSettingsList;
	typedef std::vector<BannerInput> BannerArray;
	typedef std::vector<WeightMapInput> WeightMapArray;

	GlobalAnimSettings()
	{
		setDefault();
	}
	GlobalAnimSettings& operator = (const GlobalAnimSettings& other);

	void setDefault()
	{
		m_cameraCount = 0;
		m_startFrame = 0;
		m_endFrame = -1;
		m_fileVersion = 4.0;
		m_fps = 30.0f;
		m_realFps = m_fps;
		m_fYaw = 0;
		m_fPitch = 0.0f;
		m_fRoll = 0.0f;
		m_fInteractYaw = m_fInteractPitch = m_fInteractRoll = 0;

		m_sourcefps = 30.0f;

		m_xres = 1280;
		m_yres = 960;
		m_panoXRes = 4096;
		m_panoYRes = 2048;

		m_captureType = D360::Capture::CAPTURE_DSHOW;
		
		m_stitch = true;
		m_oculus = false;
		
		m_ui = true;
		
		m_snapshotDir = "E:/";
		m_weightMapDir = "E:/";

		m_audioLag = 0;
		m_sampleFmt = 8;

		m_audioDeviceName = "";

		m_splitMins = 0.0f;

		m_videoCodec = (int)AV_CODEC_ID_H264;
		m_audioCodec = (int)AV_CODEC_ID_AAC;
		m_isGPU = false;

		m_lidarPort = -1;

		m_cameraCalibFile = "";
		m_blendingMode = Feathering;
		m_multiBandLevel = 0;
		m_fileExt = "jpg";

		m_lensType = CameraParameters::LensType_ptLens_Fullframe_Fisheye;	// Default fisheye lens
		m_fov = 240;
		
		m_leftIndices.clear();
		m_rightIndices.clear();
		m_cameraSettings.clear();
		m_weightMaps.clear();
		m_banners.clear();

		isINIfile = false;
		// Nodal shooting
		m_nodalVideoFilePathMap.clear();
		m_nodalMaskImageFilePathMap.clear();
		m_nodalVideoIndex = -1;
		m_hasNodalOfflineVideo = false;
		m_nodalSlotIndexMap.clear();
		m_videoFilePathMap.clear();
		m_foregroundSlotIndexMap.clear();
		m_haveNodalMaskImage = false;
		m_haveLiveBackground = false;

		// Streaming 
		m_wowzaServer = "";

		m_crf = 23;

		// Take Management
		m_capturePath = "C:/Capture/";

		// need to show all live cameras on UI
		m_totalCameraDevicesMap.clear();
		m_totalStereoMap.clear();
		m_totalIndexAndSelectedIndexMap.clear();

		resetLutData();
			
	}

	void setLuTData(QVariantList &vList, int colorType){

		for (int i = 0; i < LUT_COUNT; i++){
			m_lutData[colorType][i] = vList[i];
		}
	}
	QVariantList * getLutData(){
		return m_lutData;
	}


	void resetLutData(){
		m_lutData[0].clear();
		m_lutData[1].clear();
		m_lutData[2].clear();
		m_lutData[3].clear();

		for (int i = 0; i < LUT_COUNT; i++){
			m_lutData[0].append(i / 10.f);
			m_lutData[1].append(i / 10.f);
			m_lutData[2].append(i / 10.f);
			m_lutData[3].append(i / 10.f);
		}
	}

	enum Camera3DInputConfig
	{
		Stereo,
		Global
	};

	enum Format3DOutput
	{
		TopDown,
		SideBySide
	};

	enum BlendingMode
	{
		Feathering,
		MultiBandBlending,
		WeightMapVisualization,
	};

	bool isStereo() { return m_leftIndices.size() > 0 && m_rightIndices.size() > 0; }

	int     m_cameraCount;

	bool	m_ui;

	int		m_startFrame;
	int		m_endFrame;
	float	m_fileVersion;

	float	m_fps;
	float	m_realFps;
	float	m_fYaw;
	float	m_fPitch;
	float	m_fRoll;
	float	m_fInteractYaw;
	float	m_fInteractPitch;
	float	m_fInteractRoll;

	float   m_sourcefps;

	int		m_xres;
	int		m_yres;
	
	
	D360::Capture::CaptureDomain m_captureType;
	
	bool	m_stitch;
	bool	m_oculus;
	
	QString m_fileExt;
	QString m_snapshotDir;
	QString m_weightMapDir;

	// [StitchSettings]
	BlendingMode m_blendingMode;
	int m_multiBandLevel;
	int		m_panoXRes;
	int		m_panoYRes;

	// [Audio]
	QString m_audioDeviceName;		//'deviceName'
	int		m_audioLag;
	int		m_sampleFmt;

	//[LiDAR]
	int m_lidarPort;

	// [Streaming]
	float m_splitMins;		// offlineVideo's split mins, UNIT=min, if '0', no split
	QString m_wowzaServer;			//'server'			: wowza server url

	// [Nodal shooting]
	int		m_nodalVideoIndex;

	QMap<int, QString> m_nodalVideoFilePathMap;
	QMap<int, QString> m_nodalMaskImageFilePathMap;
	QMap<int, int> m_nodalSlotIndexMap;               // 'slotIndex' : 'slotIndex'	
	bool m_haveNodalMaskImage;
	bool m_haveLiveBackground;
	QMap<int, QString> m_videoFilePathMap;
	QMap<int, int> m_foregroundSlotIndexMap;          // 'slotIndex' : 'slotIndex'

	bool	m_hasNodalOfflineVideo;

	int m_videoCodec;
	int m_audioCodec;
	bool m_isGPU;

	typedef std::vector< int > CameraIndices;

	QString m_cameraCalibFile;

	CameraParameters::LensType m_lensType;
	int m_fov;

	int m_crf;

	bool useCuda;
	bool isINIfile;

	QVariantList m_lutData[4]; //0 : Gray, 1 : Red, 2 : Green, 3 : Blue

	BannerArray m_banners;
	WeightMapArray m_weightMaps;

	// [Take Management]
	QString m_capturePath;

	// need to show all live cameras on UI
	QMap<int, QString> m_totalCameraDevicesMap;
	QMap<int, int>     m_totalStereoMap;
	QMap<int, int>     m_totalIndexAndSelectedIndexMap;

	CameraIndices& getLeftIndices()
	{
		return m_leftIndices;
	}
	CameraIndices& getRightIndices()
	{
		return m_rightIndices;
	}

	int getAudioChannelCount()
	{
		int audioChannelCount = 0;
		for (int i = 0; i < m_cameraSettings.size(); i++)
		{
			if (m_cameraSettings[i].audioType != CameraInput::NoAudio)
			{
				audioChannelCount++;
			}
		}

		return audioChannelCount;
	}

	CameraSettingsList& cameraSettingsList()
	{
		return m_cameraSettings;
	}

	void refreshTempCameraSettingsList() {
		m_tempCameraSettings = m_cameraSettings;
	}

	void rollbackCameraSettingsList() {
		m_cameraSettings = m_tempCameraSettings;
	}
	
	void setCameraSettingsList(CameraSettingsList cameraList)
	{
		m_cameraSettings = cameraList;
	}

	bool detachNodalCamera(QMap<int, CameraInput>& nodalCameraMap);
	bool getOfflineCamera(QMap<int, CameraInput>& nodalCameraMap);

	QString getNodalWeightPath(int slotIndex);

protected:
	CameraIndices m_leftIndices;
	CameraIndices m_rightIndices;
	CameraSettingsList m_cameraSettings;
	CameraSettingsList m_tempCameraSettings;

	QMap<int, CameraInput> m_nodalCameraInputMap;
	CameraInput m_playbackInput; //!< Playback input information

signals:
	void fireEventBlendSettingUpdated(GlobalAnimSettings::BlendingMode mode, int level);
};

class D360Parser : public QObject
{
	Q_OBJECT
public:
	D360Parser(QObject* parent = 0);
	virtual ~D360Parser();

	void clear();

	D360Parser& operator = ( const D360Parser& other );

	//void saveINI(std::string iniFileName);
	//int parseJSON(std::string fileName);
	//int saveJSON(std::string fileName);
	//int parseXML(QString fileName);
	int parsePAC(QString fileName, GlobalAnimSettings::CameraSettingsList& cameraSettings);
	int parsePTS(QString fileName, GlobalAnimSettings::CameraSettingsList& cameraSettings);
	int parseCameraCalibrationFile(QString fileName, GlobalAnimSettings::CameraSettingsList& cameraSettings);

	int parseTemplatePAS(int camIndex, QString fileName, GlobalAnimSettings::CameraSettingsList& cameraSettings); // live camera template
	void saveTemplatePASFile(int camIndex, QString strSavePath);

	int parseTemplatePAC(QString fileName, GlobalAnimSettings::CameraSettingsList& cameraSettings, QList<int> indexList); // stitch camera template	
	void saveTemplatePACFile(QString strSavePath, QList<int> indexList);
	bool writeXMLFile(QDomDocument doc, QString filename);

	void saveTemplateFavoriteList(QString strSavePath, QList<QString> favoriteItemList);
	QList<QString> parseFavoriteTemplateXML(QString fileName); 

	GlobalAnimSettings& getGlobalAnimSettings()
	{
		return m_globalAnimSettings;
	}

	GlobalAnimSettings& getTempGlobalAnimSettings() { return m_tempGlobalAnimSettings; }

	GlobalAnimSettings& getRigTemplateGlobalAnimSettings() { return m_rigTemplateGlobalAnimSettings; }

	void resetTempGlobalAnimSettings() { 
		m_tempGlobalAnimSettings = m_globalAnimSettings;
	}

	void clearTempGlobalAnimSettings() {
		m_tempGlobalAnimSettings = GlobalAnimSettings();
	}

	//startFrame, endFrame
	float startFrame()
	{
		return m_globalAnimSettings.m_startFrame;
	}
	void setStartFrame(float fStartFrame){
		m_globalAnimSettings.m_startFrame = fStartFrame;
	}
	float endFrame()
	{
		return m_globalAnimSettings.m_endFrame;
	}

	void setEndFrame(float fEndFrame) { 
		m_globalAnimSettings.m_endFrame = fEndFrame;
	}

	//version
	float fileVersion()	{
		return m_globalAnimSettings.m_fileVersion;
	}
	//captureType
	int getCaptureType() {
		int ret = m_globalAnimSettings.m_captureType;
		return ret;
	}
	void setCaptureType(int captureType){
		m_globalAnimSettings.m_captureType = (D360::Capture::CaptureDomain)captureType;
	}
	
	//cameraCount
	int getCameraCount() { return m_globalAnimSettings.m_cameraCount; }
	void setCameraCount(int cameraCount) { m_globalAnimSettings.m_cameraCount = cameraCount; }

	//fps
	float getFps() { return m_globalAnimSettings.m_fps; }
	void setFps(float fps) { m_globalAnimSettings.m_fps = fps; }

	//fps
	float getSourceFps() { return m_globalAnimSettings.m_sourcefps; }
	void setSourceFps(float sourcefps) { m_globalAnimSettings.m_sourcefps = sourcefps; }

	//input resolution
	int getXres() { return m_globalAnimSettings.m_xres; }
	void setXres(int xres) { m_globalAnimSettings.m_xres = xres; }
	int getYres() { return m_globalAnimSettings.m_yres; }
	void setYres(int yres) { m_globalAnimSettings.m_yres = yres; }

	//cameraClib
	QString getCameraCalib() { return m_globalAnimSettings.m_cameraCalibFile; }
	void setCameraCalib(QString cameraCalib) {
		m_globalAnimSettings.m_cameraCalibFile = cameraCalib; 
	}

	//sample Fmt
	int getSampleFmt() { return m_globalAnimSettings.m_sampleFmt; }
	void setSampleFmt(int sampleFmt) { m_globalAnimSettings.m_sampleFmt = sampleFmt; }

	//lag
	int getLag() { return m_globalAnimSettings.m_audioLag; }
	void setLag(int lag) { m_globalAnimSettings.m_audioLag = lag; }

	//output resolution
	int getPanoXres() { return m_globalAnimSettings.m_panoXRes; }
	void setPanoXres(int panoXres) { m_globalAnimSettings.m_panoXRes = panoXres; }
	int getPanoYres() { return m_globalAnimSettings.m_panoYRes; }
	void setPanoYres(int panoYres) { m_globalAnimSettings.m_panoXRes = panoYres; }

	//device Count
	int getDeviceCount() { return m_globalAnimSettings.m_cameraCount; }
	void setDeviceCount(int deviceCount) { }
	//yaw
	float  getYaw(int deviceNum);
	void setYaw(float yaw, int deviceNum);
	//pitch
	float getPitch(int deviceNum);
	void setPitch(float pitch, int deviceNum);
	
	float getRoll(int deviceNum);
	void setRoll(float roll, int deviceNum);

	// Blending parameters
	float  getLeft(int iDeviceNum);
	void setLeft(float fLeft, int iDeviceNum);
	float getRight(int iDeviceNum);
	void setRight(float fRight, int iDeviceNum);
	float getTop(int iDeviceNum);
	void setTop(float fTop, int iDeviceNum);
	float getBottom(int iDevicNum);
	void setBottom(float fBottom, int iDeviceNum);
	
	// gain
	int getCameraGain(int iDeviceNum);
	void setCameraGain(int iGain, int iDeviceNum);

	//stereo
	int getStereoType(int iDeviceNum);
	void setStereoType(int iStereoType, int iDeviceNum);
	void setTempCameraSettings();
	void resetCameraSettings();

	//audioType
	void setAudioType(int iAudioType, int iDeviceNum);

	void setTempStereoType(int, int);
	int getTempStereoType(int iDeviceNum);
	void clearTempStereoType() { m_tempStereoList.clear(); }
	QMap<int, int> getTempStereoTypeList() { return m_tempStereoList; }
	void setRigTemplateCameraSettings() { m_rigTemplateGlobalAnimSettings = m_globalAnimSettings; }

	void setTempImagePath(int, QString);
	QString getTempImagePath(int iDeviceNum);
	void setTempImagePrefix(int, QString);
	QString getTempImagePrefix(int iDeviceNum);
	void setTempImageExt(int, QString);
	QString getTempImageExt(int iDeviceNum);
	void setTempAudioSettings(int, int);
	int getTempAudioSettings(int iDeviceNum);
	void clearStereoList() { m_tempStereoList.clear(); m_tempAudioList.clear(); }
public slots:	
	bool parseINI(QString iniFileName, bool isRigTemplate = false);
	void saveINI(QString l3dFilename, QString iniFileName, bool isRigTemplate = false);
	void initTemplateVideo(QMap<int, QString> videoPathList);
	void initTemplateCamera(QMap<int, QString> cameraNameList, QMap<int, QString> audioNameList);
	void initTemplateImage(QMap<int, QString> imagePathList);
signals:

protected:
	GlobalAnimSettings m_globalAnimSettings;
	GlobalAnimSettings m_tempGlobalAnimSettings;
	GlobalAnimSettings::CameraSettingsList m_originCameraSettings;
	GlobalAnimSettings m_rigTemplateGlobalAnimSettings;
	GlobalAnimSettings::CameraSettingsList m_tempCameraList;
	QMap<int, int> m_tempStereoList;
	QMap<int, int> m_tempAudioList;
	QMap<int, QString> m_tempPathList;
	QMap<int, QString> m_tempPrefixList;
	QMap<int, QString> m_tempExtList;
private:

};
#pragma once
