#ifndef STREAMPROCESS_H
#define STREAMPROCESS_H

#include <QObject>
#include <QTime>
#include <QThread>
#include <QMutex>
#include <QQueue>

#include <Structures.h>
#include <Buffer.h>

#include "StreamFfmpeg.hpp"
#include "SharedImageBuffer.h"

// This class is for take recording and rtmp streaming.
class StreamProcess : public QObject
{
	Q_OBJECT

public:
	StreamProcess(SharedImageBuffer* sharedImageBuffer, bool toFile, QObject* parent = NULL);
	virtual ~StreamProcess();

	virtual bool	initialize(bool toFile, QString outFileName, int width, int height, int fps,
		int channelCount = 0, AVSampleFormat sampleFmt = AV_SAMPLE_FMT_S16, 
		int srcSampleRate = 0, int sampleRate = 0, int audioLag = 0, 
		int videoCodec = (int)AV_CODEC_ID_H264, int audioCodec = (int)AV_CODEC_ID_AAC, int crf = 23);
	virtual bool	initializeStream(int nSerialNumber); // ONLY for replay function	

	void	stopStreamThread();
	void	waitForFinish();
	bool	isFinished();
	void	setFinished() { m_finished = true; };
	bool	disconnect();

	void	playAndPause(bool isPause);
	bool	isOpened() { return m_isOpened; }

	bool	isTakeRecording() { return m_isFile; }
	bool	isWebRTC() { return m_iswebRTC; }

protected:
	void init();
	void updateFPS(int);
	void setMain(QObject* main) { m_Main = main; }

	QString m_Name;
	QTime t;
	QMutex doExitMutex;
	QMutex finishMutex;
	QWaitCondition finishWC;
	bool m_finished;

	QQueue<int> fps;

	struct ThreadStatisticsData statsData;
	bool m_isOpened;
	bool doExit;
	int captureTime;
	int sampleNumber;
	int fpsSum;

	QThread* streamingThreadInstance;

	int m_audioInputCount;
	int m_videoInputCount;
	int m_lidarInputCount;
	int m_audioProcessedCount;
	int m_videoProcessedCount;
	int m_lidarProcessedCount;
	StreamFfmpeg m_Stream;
	StreamFfmpeg m_LiDARStream; // ???
	//RawImagePtr m_Panorama;
	void * m_audioFrame;
	//QMap<int, QList<void*>>	m_audioFrames;
	QMap<int, void*>	m_audioFrames;
	QMap<int, void*>	m_audioReadyFrames;
	ImageBufferData		m_LiDARFrame;
	int m_audioChannelCount;
	//QList<QImage> m_Panoramas;
	unsigned char* m_Panorama;
		
	bool m_toFile;
	QString m_outFileName;
	int m_fps;
	int m_width;
	int m_height;
	int m_channelCount;
	AVSampleFormat m_sampleFmt;
	int m_srcSampleRate;
	int m_sampleRate;
	int m_audioLag;
	int m_videoCodec;
	int m_audioCodec;
	int m_crf;

	// Saves # of hdd video file. 0 means first or single video, above 0 means it's subsequent files.
	// Need to signal this index to main window so that capture threads shouldn't be run twice.
	int m_nCurrentSplitSerialNumber;

	bool m_isFile;
	SharedImageBuffer* m_sharedImageBuffer;
	QMutex videoFrameMutex;
	QMutex audioFrameMutex;		
	QMutex lidarFrameMutex;

	QMutex doPauseMutex;
	bool doPause;

	QObject* m_Main;

	bool m_iswebRTC;

protected:
	void run();

public slots:
	void process();
	void streamPanorama(unsigned char* panorama);
	void streamAudio(int devNum, void* audioFrame);
	void streamLiDAR( ImageBufferData& frame );
	QMap<int, void*> getAudioFrameSeq();

signals:
	void updateStatisticsInGUI(struct ThreadStatisticsData);
	//void newFrame(const QPixmap &frame, int frameNum);
	void finished(int type, QString msg, int id);
	void started(int type, QString msg, int id);
};

// This class is for take recording and webRTC streaming.
class webRTC_StreamProcess : public StreamProcess
{
Q_OBJECT

public:
	webRTC_StreamProcess(SharedImageBuffer* sharedImageBuffer, bool toFile, QObject* parent = NULL);

	virtual bool	initializeStream(int nSerialNumber); // ONLY for replay function	
};
#endif // STREAMPROCESS_H