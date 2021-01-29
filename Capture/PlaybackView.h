#pragma once

//
// Local
//
#include "PlaybackThread.h"
#include "Structures.h"
#include "PlaybackStitcher.h"
#include "SharedImageBuffer.h"

class PlaybackModule : public QObject
{
	Q_OBJECT

public:
	explicit PlaybackModule(SharedImageBuffer *sharedImageBuffer, QObject* main = NULL);
	virtual ~PlaybackModule();

	void qquit();
	void stopPlaybackThread();
	void startThreads(bool isReplay = false);

	virtual bool connectToPlayback(int width, int height, D360::Capture::CaptureDomain cameraType);
	IAudioThread * getAudioThread();
	PlaybackThread * getPlaybackThread();
	bool isConnected() { return isPlaybackConnected; }

protected:

	QObject* m_Main;
	PlaybackThread  *playbackThread;
	AudioThread		*audioThread;
	AudioInput		*mic;

	SharedImageBuffer *sharedImageBuffer;

	bool isPlaybackConnected;
	QString m_Name;

	QThread* playbackThreadInstance;

signals:
	void newImageProcessingFlags(struct ImageProcessingFlags imageProcessingFlags);
};