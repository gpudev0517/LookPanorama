
#pragma once

// Qt
#include <QHash>
#include <QSet>
#include <QWaitCondition>
#include <QMutex>
#include "ImageBuffer.h"
#include <QImage>

// Local
#include "D360Parser.h"
#include "Buffer.h"

#define D360_FILEDEVICESTART 100


class GlobalState
{
public:
	float m_curFrame;
};

class D360Stitcher;
class PlaybackStitcher;
class StreamProcess;

class SharedImageBuffer : public QObject
{
	Q_OBJECT
public:
	typedef ImageBufferData ImageDataPtr;

	SharedImageBuffer();
	virtual ~SharedImageBuffer();
	void initialize();
	void initializeForReplay();
	void setSeekFrames(int nFrames);
	void add(int deviceNumber, bool sync = false);
	void setPlaybackRawImage(ImageDataPtr image);
	void setRawImage(int deviceNumber, ImageDataPtr image);
	ImageDataPtr getRawImage(int deviceNumber);

	void addState(int deviceNumber, GlobalState& state, bool sync = false);
	GlobalState& getState(int deviceNumber)
	{
		return m_globalStates[deviceNumber];
	}

	void removeByDeviceNumber(int deviceNumber);
	void sync(int deviceNumber);
	void wakeAll();
	void setSyncEnabled(bool enable);
	void setViewSync(int deviceNumber, bool enable);
	bool isSyncEnabledForDeviceNumber(int deviceNumber);
	bool getSyncEnabled();
	int	 getSyncedCameraCount();
	int getFirstAvailableViewId();

	void setGlobalAnimSettings(GlobalAnimSettings* globalAnimSettings)
	{
		m_globalAnimSettings = globalAnimSettings;
	}

	GlobalAnimSettings* getGlobalAnimSettings()
	{
		return m_globalAnimSettings;
	}

	std::shared_ptr< D360Stitcher > getStitcher()
	{
		return m_stitcher;
	}

	void setStitcher(std::shared_ptr< D360Stitcher > stitcher);

	std::shared_ptr< PlaybackStitcher > getPlaybackStitcher()
	{
		return m_playbackStitcher;
	}

	void setPlaybackStitcher(std::shared_ptr< PlaybackStitcher > stitcher);

	bool syncForVideoProcessing(int videoFrameId);
	void syncForAudioProcessing(int audioFrameId = 0);
	void wakeForVideoProcessing(int videoFrameId);
	void wakeForAudioProcessing(int audioFrameId);

	bool syncForVideoPlayback(int videoFrameId);
	void syncForAudioPlayback(int audioFrameId = 0);
	void wakeForVideoPlayback(int videoFrameId);
	void wakeForAudioPlayback(int audioFrameId);

	void waitStitcher();
	void wakeStitcher();

	void setCaptureFinalizing();
	bool isCaptureFinalizing();

	void lockPlaybackBuffer();
	void unlockPlaybackBuffer();

	void addCamera(int cameraIndex);
	void removeCamera(int cameraIndex);
	void lockIncomingBuffer(int cameraIndex);
	void unlockIncomingBuffer(int cameraIndex);
	void lockOculus();
	void unlockOculus();

	void setLiveGrabber(int camIndex);
	int getLiveGrabber();

	void setStreamer(StreamProcess* streamer) { m_streamer = streamer; }
	StreamProcess* getStreamer() { return m_streamer; }

	void setWeightMapEditEnabled(bool isEnabled);
	bool isWeightMapEditEnabled();

	void selectView(int viewIndex1, int viewIndex2);
	int getSelectedView1();
	int getSelectedView2();
private:
	StreamProcess* m_streamer;
	ImageDataPtr playbackRawImage;
	std::map<int, ImageDataPtr> rawImages;

	QSet< int > syncSet;
	QWaitCondition wc;
	QMutex mutex;
	int nArrived;
	bool doSync;

	// Offline processing
	QMutex videoMutex;
	QWaitCondition videoWCondition;
	int videoProcessedId;
	int videoPlaybackId;

	QMutex audioMutex;
	QWaitCondition audioWCondition;
	int audioCapturedCount;
	int audioProcessedId;
	int audioPlaybackId;
	
	bool isFinalizing; // camera threads use this flag to know if others are closing already

	QMutex stitcherMutex;
	QWaitCondition stitcherWC;

	QMap<int, QMutex*> incomingBufferMutex;
	QMutex playbackBufferMutex;
	QMutex oculusMutex;

	int liveGrabIndex;

	GlobalAnimSettings* m_globalAnimSettings;
	QHash< int, GlobalState> m_globalStates;

	std::shared_ptr< D360Stitcher > m_stitcher;
	std::shared_ptr< PlaybackStitcher > m_playbackStitcher;

	// Weight map status
	bool weightEditEnabled;

	// Current view for seam
	// -1: all disable, 0: view all camera seams, 1~n: view single seam
	int selectedViewIndex1;
	int selectedViewIndex2;

signals:
	void fireEventWeightMapEditEnabled(bool isEnabled);
	void fireEventViewSelected(int viewIndex1, int viewIndex2);
};
