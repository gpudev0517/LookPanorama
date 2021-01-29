#include "GlobalAnimSettings.h"


GlobalAnimSettings& GlobalAnimSettings::operator = (const GlobalAnimSettings& other)
{
	try
	{
		m_cameraCount = other.m_cameraCount;

		m_startFrame = other.m_startFrame;
		m_endFrame = other.m_endFrame;
		m_fileVersion = other.m_fileVersion;
		m_exp = other.m_exp;
		m_gain = other.m_gain;
		m_fps = other.m_fps;

		m_3dOutput = other.m_3dOutput;
		m_3dCameraInputConfig = other.m_3dCameraInputConfig;
		m_leftIndices = other.m_leftIndices;
		m_rightIndices = other.m_rightIndices;

		m_xres = other.m_xres;
		m_yres = other.m_yres;
		m_panoXRes = other.m_panoXRes;
		m_panoYRes = other.m_panoYRes;
		m_audioLag = other.m_audioLag;
		m_sampleFmt = other.m_sampleFmt;
		m_sampleRate = other.m_sampleRate;
		m_verticallyMirrored = other.m_verticallyMirrored;

		m_playbackfps = other.m_playbackfps;

		m_triggerMode = other.m_triggerMode;
		m_triggerSource = other.m_triggerSource;
		m_acqusitionMode = other.m_acqusitionMode;

		m_saveCaptureFrames = other.m_saveCaptureFrames;
		m_record = other.m_record;
		m_captureMode = other.m_captureMode;
		m_captureType = other.m_captureType;
		m_bs = other.m_bs;

		m_capture = other.m_capture;
		m_debayer = other.m_debayer;
		m_saveDebayerFrames = other.m_saveDebayerFrames;

		m_stitch = other.m_stitch;
		m_oculus = other.m_oculus;
		m_saveStitchFrames = other.m_saveStitchFrames;

		m_display = other.m_display;
		m_ui = other.m_ui;
		m_activeCam = other.m_activeCam;
		m_compression = other.m_compression;
		m_compressionQuality = other.m_compressionQuality;

		m_snapshotDir = other.m_snapshotDir;

		m_audioDeviceName = other.m_audioDeviceName;

		m_offlineVideo = other.m_offlineVideo;
		m_videoCodec = other.m_videoCodec;
		m_audioCodec = other.m_audioCodec;
		m_wowzaServer = other.m_wowzaServer;

		m_cameraCalibFile = other.m_cameraCalibFile;

		m_cameraSettings = other.m_cameraSettings;
		m_unwarp = other.m_unwarp;
	}
	catch (...)
	{

	}


	return *this;
}