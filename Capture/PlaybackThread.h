
#ifndef PLAYBACKTHREAD_H
#define PLAYBACKTHREAD_H


#include <thread>

// Qt
#include <QPixmap>
#include <QtCore/QTime>
#include <QtCore/QThread>


#include "SharedImageBuffer.h"
#include "Config.h"
#include "Structures.h"
#include "Capture.h"
#include "AudioThread.h"
#include "StreamProcess.h"

class ImageBuffer;

class PlaybackThread : public IAudioThread
{
    Q_OBJECT

    public:
		PlaybackThread(SharedImageBuffer *sharedImageBuffer, D360::Capture::CaptureDomain capType, int width, int height);
		virtual ~PlaybackThread();

		void	init();
        void	stop();
		void	playAndPause(bool isPause);
		void	setSeekFrames(int nFrames, bool bIsFirstFrame = false);
        bool	connect();
        bool	disconnect();
		bool	reconnect();
        bool	isConnected();
        int		getInputSourceWidth();
        int		getInputSourceHeight();

		virtual AudioInput * getMic();

		void forceFinish();

		void waitForFinish();

		bool isFinished();

    private:
        void updateFPS( int );
		void waitCapture();
		void wakeCapture();

		QString m_Name;
        SharedImageBuffer *sharedImageBuffer;
		D360::Capture* cap;

		D360::Capture::CaptureDomain m_captureType;

        SharedImageBuffer::ImageDataPtr m_grabbedFrame;
        QTime t;
        QMutex doStopMutex;
		QMutex doPauseMutex;
		QMutex doSeekMutex;
        QQueue<int> fps;

		QMutex finishMutex;
		QWaitCondition finishWC;
		bool m_finished;
		bool m_isCanGrab;

		QMutex pauseWCMutex;
		QWaitCondition pauseWC;

		bool m_isReplay;
		bool m_bIsFirstFrame;

        struct ThreadStatisticsData statsData;
        bool doStop;		
		bool doPause;
        int captureTime;
        int sampleNumber;
        int fpsSum;
        int width;
        int height;

		float m_captureExp;
		float m_captureGain;
		float m_captureFPS;

    protected:
        void run();

	public slots:
			void process();

    signals:
        void updateStatisticsInGUI( struct ThreadStatisticsData );
		void finished(int type, QString msg, int id);
		void started(int type, QString msg, int id);
		void report(int type, QString msg, int id);
		void firstFrameCaptured(int);
};

#endif // CAPTURETHREAD_H
