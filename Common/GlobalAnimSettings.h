#pragma once

#include <vector>

#include <QMap>
#include <QDebug>

#include "libavcodec/avcodec.h"

#include "Structures.h"
#include "CaptureXimea.h"

class GlobalAnimSettings : public QObject
{
	Q_OBJECT
	Q_PROPERTY(float exposure READ getExposure WRITE setExposure NOTIFY stateChanged)
	Q_PROPERTY(int captureType READ getCaptureType WRITE setCaptureType NOTIFY stateChanged)
	Q_PROPERTY(int fps READ getFps WRITE setFps NOTIFY stateChanged)
	Q_PROPERTY(int gain READ getGain WRITE setGain NOTIFY stateChanged)
	Q_PROPERTY(int xres READ getXres WRITE setXres NOTIFY stateChanged)
	Q_PROPERTY(int yres READ getYres WRITE setYres NOTIFY stateChanged)
	Q_PROPERTY(int compressionMode READ getCompressionMode WRITE setCompressionMode NOTIFY stateChanged)
	Q_PROPERTY(int compressionQuality READ getCompressionQuality WRITE setCompressionQuality NOTIFY stateChanged)
	Q_PROPERTY(int captureMode READ getCaptureMode WRITE setCaptureMode NOTIFY stateChanged)
	Q_PROPERTY(QString cameraCalib READ getCameraCalib WRITE setCameraCalib NOTIFY stateChanged)
	Q_PROPERTY(int sampleFmt READ getSampleFmt WRITE setSampleFmt NOTIFY stateChanged)
	Q_PROPERTY(int sampleRate READ getSampleRate WRITE setSampleRate NOTIFY stateChanged)
	Q_PROPERTY(int lag READ getLag WRITE setLag NOTIFY stateChanged)
	Q_PROPERTY(int panoXres READ getPanoXres WRITE setPanoXres NOTIFY stateChanged)
	Q_PROPERTY(int panoYres READ getPanoYres WRITE setPanoYres NOTIFY stateChanged)
	Q_PROPERTY(int bufferSize READ getBufferSize WRITE setBufferSize NOTIFY stateChanged)
	Q_PROPERTY(int streamMode READ getStreamMode WRITE setStreamMode NOTIFY stateChanged)
	Q_PROPERTY(QString streamURL READ getStreamURL WRITE setStreamURL NOTIFY stateChanged)
	Q_PROPERTY(int deviceCount READ getDeviceCount WRITE setDeviceCount NOTIFY stateChanged)
	
signals:
	void stateChanged();
public slots:
	void set(QObject* settings) {
		operator = (*(qobject_cast<GlobalAnimSettings*>(settings)));
	}
	QObject* get() { return (QObject*)this; }
	//captureType
	int getCaptureType() {
		switch (m_captureType)
		{
		case D360::Capture::CAPTURE_DSHOW:
			return 0;
			break;
		case D360::Capture::CAPTURE_XIMEA:
			return 1;
			break;
		case D360::Capture::CAPTURE_VIDEO:
			return 2;
			break;
		case D360::Capture::CAPTURE_FILE:
			return 3;
			break;
		default:
			break;
		}
		return m_captureType; }
	void setCaptureType(int captureType){
		switch (captureType)
		{
		case 0: m_captureType = D360::Capture::CAPTURE_DSHOW; break;
		case 1: m_captureType = D360::Capture::CAPTURE_XIMEA; break;
		case 2: m_captureType = D360::Capture::CAPTURE_VIDEO; break;
		case 3: m_captureType = D360::Capture::CAPTURE_FILE; break;
		default:
			break;
		}
	}
	//exposure
	float getExposure() { return m_exp; }
	void setExposure(float exposure) {
		m_exp = exposure; 
	}
	//cameraCount
	int getCameraCount() { return m_cameraCount; }
	void setCameraCount(int cameraCount) { m_cameraCount = cameraCount; }
	//fps
	int getFps() { return m_fps; }
	void setFps(int fps) { m_fps = fps; }
	//gain
	int getGain() { return m_gain; }
	void setGain(int gain) { 
		m_gain = gain; }
	//input resolution
	int getXres() { return m_xres; }
	void setXres(int xres) { m_xres = xres; }
	int getYres() { return m_yres; }
	void setYres(int yres) { m_yres = yres; }
	//compress
	int getCompressionMode() {
		switch (m_compression)
		{
		case 4:
			return 0;
			break;
		case 3:
			return 1;
			break;
		case 6:
			return 2;
			break;
		case 1:
			return 3;
			break;
		case 17:
			return 4;
			break;
		default:
			break;
		}
		return m_compression;
		
	}
	void setCompressionMode(int compressionMode){ m_compression = compressionMode; }
	//compression Quality
	int getCompressionQuality() { return m_compressionQuality; }
	void  setCompressionQuality(int compressionQuality) { m_compressionQuality = compressionQuality; }
	//capture Mode
	int getCaptureMode() {
		switch (m_captureMode)
		{
		case CaptureXimea::FRAMERATE:
			return 0;
			break;
		case CaptureXimea::FREERUN:
			return 1;
			break;
		case CaptureXimea::RISINGEDGE:
		    return 2;
			break;
		default:
			break;
		}
		return m_captureMode; 
	}
	void setCaptureMode(int captureMode) {
		switch (captureMode)
		{
		case 0: m_captureMode = CaptureXimea::FRAMERATE; break;
		case 1: m_captureMode = CaptureXimea::FREERUN; break;
		case 2: m_captureMode = CaptureXimea::RISINGEDGE; break;
		default:
			break;
		}
	}
	//cameraClib
	QString getCameraCalib() { return m_cameraCalibFile; }
	void setCameraCalib(QString cameraCalib) { m_cameraCalibFile = cameraCalib; }
	//sample Fmt
	int getSampleFmt() { return m_sampleFmt; }
	void setSampleFmt(int sampleFmt) { m_sampleFmt = sampleFmt; }
	//sample Rate
	int getSampleRate() { return m_sampleRate; }
	void setSampleRate(int sampleRate) { m_sampleRate = sampleRate; }
	//lag
	int getLag() { return m_audioLag; }
	void setLag(int lag) { m_audioLag = lag; }
	//output resolution
	int getPanoXres() { return m_panoXRes; }
	void setPanoXres(int panoXres) { m_panoXRes = panoXres; }
	int getPanoYres() { return m_panoYRes; }
	void setPanoYres(int panoYres) { m_panoXRes = panoYres; }
	//bufferSize
	int getBufferSize() { return m_bs; }
	void setBufferSize(int bufferSize) { m_bs = bufferSize; }
	// stream
	int getStreamMode() {
		if (m_wowzaServer.trimmed().isEmpty())	return 1;
		return 0;
	}
	void setStreamMode(int mode) {
		// Not implemented.
	}
	QString getStreamURL() {
		if (m_wowzaServer.trimmed().isEmpty())	return m_offlineVideo;
		return m_wowzaServer;
	}
	void setStreamURL(QString streamURL) {
		m_offlineVideo = streamURL;
	}
	//device Count
	int getDeviceCount() { return m_cameraCount; }
	void setDeviceCount(int deviceCount) { }

public:

	typedef std::vector<CameraInput> CameraSettingsList;


	GlobalAnimSettings()
	{
		setDefault();
	}
	GlobalAnimSettings& operator = (const GlobalAnimSettings& other);

	void setDefault()
	{
		m_cameraCount = 0;
		m_startFrame = -1;
		m_endFrame = -1;
		m_fileVersion = 1.0;
		m_exp = 1000.0f;
		m_gain = 1.0f;
		m_fps = 60.0f;

		m_3dCameraInputConfig = 0;
		m_3dOutput = 0;

		m_playbackfps = 30.0f;

		m_triggerMode = 0;
		m_triggerSource = 0;
		m_acqusitionMode = 0;

		m_xres = 4096;
		m_yres = 2048;

		m_saveCaptureFrames = false;
		m_record = false;
		m_captureMode = CaptureXimea::RISINGEDGE;
		m_captureType = D360::Capture::CAPTURE_DSHOW;
		m_bs = 100;
		m_verticallyMirrored = false;

		m_capture = true;
		m_debayer = false;
		m_saveDebayerFrames = false;

		m_unwarp = false;
		m_stitch = false;
		m_oculus = false;
		m_saveStitchFrames = false;

		m_display = true;
		m_ui = true;
		m_activeCam = 0;
		m_compression = 0;
		m_compressionQuality = 100;

		m_snapshotDir = "D:/";

		m_audioLag = 0;
		m_sampleFmt = 1;
		m_sampleRate = 44100;

		m_audioDeviceName = "";

		m_offlineVideo = "";
		m_videoCodec = (int)AV_CODEC_ID_H264;
		m_audioCodec = (int)AV_CODEC_ID_AAC;
		m_isGPU = false;
		m_wowzaServer = "";

		m_cameraCalibFile = "";
		m_cameraSettings.clear();

		return;
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

	bool isStereo() { return m_leftIndices.size() > 0 || m_rightIndices.size() > 0; }

	int     m_cameraCount;

	bool	m_display;
	bool	m_ui;

	float	m_startFrame;
	float	m_endFrame;
	float	m_fileVersion;

	int		m_3dCameraInputConfig; // 0: none, 1:stereo, 2:global
	int		m_3dOutput; // 0: normal, 1: topdown, 2:sidebyside

	bool    m_capture;
	float	m_exp;
	float	m_gain;
	float	m_fps;

	float   m_playbackfps;

	int		m_triggerMode;
	int		m_triggerSource;
	int		m_acqusitionMode;
	int		m_sensorDataDepth;
	int		m_sensorOutputBitDepth;
	int		m_xres;
	int		m_yres;
	int		m_panoXRes;
	int		m_panoYRes;
	int		m_audioLag;
	int		m_sampleFmt;
	int		m_sampleRate;
	int		m_verticallyMirrored;

	bool	m_saveCaptureFrames;
	bool    m_record;
	int     m_captureMode;
	int		m_captureType;
	int		m_bs;

	bool	m_debayer;
	bool	m_saveDebayerFrames;

	bool	m_unwarp;
	bool	m_saveUnwarpFrames;

	bool	m_stitch;
	bool	m_oculus;

	bool	m_saveStitchFrames;
	int		m_activeCam;
	int     m_compression;

	int     m_compressionQuality;

	QString m_snapshotDir;

	// [Audio]
	QString m_audioDeviceName;		//'deviceName'

	// [Streaming]
	QString m_offlineVideo;			//'offlineVideo'	: video file name for offline stream saving. if '', skip saving video
	int m_videoCodec;
	int m_audioCodec;
	bool m_isGPU;
	QString m_wowzaServer;			//'server'			: wowza server url 

	typedef std::vector< int > CameraIndices;

	QString m_cameraCalibFile; // left by default
	QString m_cameraCalibFileSecondary; // right

	CameraIndices& getLeftIndices()
	{
		return m_leftIndices;
	}
	CameraIndices& getRightIndices()
	{
		return m_rightIndices;
	}

	void setAudioChannel(int deviceNumber, int channel = 0)
	{
		m_audioChannels[deviceNumber] = channel;	// 0:Stereo, 1:Left, 2:Right
	}
	void setAudioChannels(QMap<int, int> audioChannels) { m_audioChannels = audioChannels; }

	QMap<int, int> getAudioChannels() { return m_audioChannels; }
	int getAudioChannelCount() {
		int channels = 0;
		QMapIterator<int, int> i(m_audioChannels);
		while (i.hasNext()) {
			i.next();
			if (i.value() == 0)	// Stereo
				channels += 2;
			else if (i.value() == 1 || i.value() == 2)	// Left/Right
				channels++;
		}
		return channels;
	}
	void clearAudioChannels() { m_audioChannels.clear(); }

	CameraSettingsList& cameraSettingsList()
	{
		return m_cameraSettings;
	}

protected:
	CameraIndices m_leftIndices;
	CameraIndices m_rightIndices;
	QMap< int, int > m_audioChannels;
	CameraSettingsList m_cameraSettings;
	CameraSettingsList m_cameraSettingsSecondary;
};