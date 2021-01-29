#ifndef AUDIOTHREAD_H
#define AUDIOTHREAD_H


#include <thread>

// Qt
#include <QtCore/QTime>
#include <QtCore/QThread>
// Local

#include "SharedImageBuffer.h"
#include "Config.h"
#include "Structures.h"

#include "CaptureAudio.h"
#include "StreamProcess.h"


class ImageBuffer;

class IAudioThread : public QObject
{
	Q_OBJECT

public:
	virtual AudioInput*		getMic() = 0;

signals:
	void newAudioFrameReady(void*);
};

class AudioThread : public IAudioThread
{
    Q_OBJECT

    public:
		AudioThread(QObject* main = NULL);
		virtual ~AudioThread();

		// thread
		void	initialize(SharedImageBuffer *sharedImageBuffer, int deviceIndex = -1);
		void	qquit();

		void	stopAudioThread();
		void	waitForFinish();
		bool	isFinished();
		
		void	start();
		void	stop();
		void	pause();

        bool	connect();
        bool	disconnect();
        bool	isConnected();

		void startThread();

		virtual AudioInput * getMic();

    private:
        void updateFPS( int );

        SharedImageBuffer *sharedImageBuffer;
		AudioMicInput* cap;

		QThread* audioThreadInstance;
        QTime t;

		QMutex doExitMutex;
		bool doExit;
		QMutex finishMutex;
		QWaitCondition finishWC;

		QMutex doPauseMutex;
		QWaitCondition pausehWC;
		
		QQueue<int> fps;
		StreamProcess* m_streamer;

		bool	m_finished;

        struct ThreadStatisticsData statsData;
        int captureTime;
        int sampleNumber;
        int fpsSum;
        int m_deviceNumber;
		QString m_audioDeviceName;
		QString m_Name;
		QObject* m_Main;
        
    protected:
        void run();

	public slots:
			void process();

    signals:
        void updateStatisticsInGUI( struct ThreadStatisticsData );
		void finished(int type, QString msg, int id);
		void started(int type, QString msg, int id);
};

#endif // AUDIOTHREAD_H
