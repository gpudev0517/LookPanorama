#ifndef CALIBPROCESS_H
#define CALIBPROCESS_H

#include <QProcess>
#include <QMap>

#include "Structures.h"
#include "SnapshotDetector.h"
#include "SelfCalibrator.h"
#include "SharedImageBuffer.h"

#define CALIB_CMD			QString("PanoramaTools.bat")
#define CALIB_WORK_PATH		QString("./PanoramaTools/")
#define CALIB_RESULT_FILE   "complete.pto"
#define CALIB_RESULT_FILE_2 "calib_result.pto"
#define START_SYM           "#"
#define VALUE_SYM           "-"
#define VALUE_EXP           "#hugin_"
#define COMMENT_EXP         "# "
#define ROOT_BLOCK          "Root"
#define IMAGE_VAL           "image lines"
#define CP_VAL              "control points"
#define OPT_VAL             "optimized"
#define CAMERA_EXP          "i "
#define CP_EXP              "c "
#define OPT_EXP             "v "

typedef enum {
    NONE_VAL = -1,
    BOOL_VAL,
    NUMBER_VAL,
    STRING_VAL,
    STRUCTURE_VAL,
} TYPEOFVAL;

typedef struct {
    TYPEOFVAL type;
    QStringList value;
    int nbVal;
    QMap<QString, float> camParams;
} PTOVAL;

typedef struct {
	int point1;
	float x1;
	float y1;
	int point2;
	float x2;
	float y2;
} CPOINT;

class CalibProcess : public QObject
{
    Q_OBJECT
public:
    CalibProcess(QString command = "", QObject * parent = 0);
    void setPath(QString path) {
        if (!path.isNull() && !path.isEmpty())
        m_Process.setWorkingDirectory(path);
    }
	bool calibrate(QString command = "", bool isFinal = false);
	void setCommand(QString command);
	QString getCommand();
    void addArg(QString arg);
	bool parsePTO(QString filename, bool primary, QMap<QString, PTOVAL>& paramMap);
	QMap<QString, PTOVAL> getCalibParams() { return m_ptoConfig; }
	bool isError() { return m_isError; }
	bool isFinished() { return m_isFinished; }
	void initialize();

private:
    QString m_Name;
    QProcess m_Process;
    QString m_Command;
    QString m_Output;
    QMap<QString, PTOVAL>   m_ptoConfig;
    QMap<QString, QString>  m_pacPTS;
	bool m_isError;
	bool m_isFinished;
	bool m_isFinal;

    int processLine(QString line);

public slots:
    void readOutput();
    void finishedCalib();
	void errorProcess(QProcess::ProcessError error);

signals:
	void finishedSignal(int);
};


class SingleCameraCalibProcess : public QObject
{
	Q_OBJECT
public:
	SingleCameraCalibProcess();
	virtual ~SingleCameraCalibProcess();

	void setParameter(int camIndex,
		CameraParameters::LensType lensType,
		int boardSizeW, int boardSizeH,
		int snapshotNumber = 10);
	void setSharedImageBuffer(SharedImageBuffer *sharedImageBuffer);
	void initialize();

	void startCapture();
	void stopCapture();
	bool isReadyToCalibrate();
	bool calibrate();
	void apply();

	void startThread();
	void stopThread();
	void waitForFinish();
	void takeSnapshot();

private:
	CSnapshotDetector detector;
	CSelfCalibrator solver;
	SharedImageBuffer *sharedImageBuffer;
	int nSnapshots;

	// Device
	int camIndex;

	// Lens type to calibrate
	int maxSnapshotCount;

	unsigned char* previewBuffer;

	// status
	enum SelfCalibStatus
	{
		None,
		Capturing,
		ReadyToCalibrate,
		FindCorners,
		Calibrating,
		Finished,
		Failed,
	};
	SelfCalibStatus calibStatus;
	std::string statusString;

	void renderStatus();

	// thread data
	QMutex threadMutex;
	Mat srcImg;
	bool isSrcReady;
	bool isOnCalibration;

	// threads
	void setupThread();
	void setFinished();
	bool isFinished();
	void releaseThread();

	QMutex finishMutex;
	QWaitCondition finishWC;
	bool m_finished;

	QThread* m_threadInstance;
	bool doStop;
	QMutex doStopMutex;

	double m_avgErr;

protected:
	void run();

public slots:
	// This event is triggered when capture started, and selected camera's frame
	// is ready.
	void onFrame(unsigned char* buffer, int width, int height);

	// Triggered when new snapshot found
	void onSnapshot(int snapshotCount);
	void onStrengthRatio(float strengthRatio);
	void onStrengthDrawColor(QString color);

	void process();
	void qquit();

signals:
	void fireSnapshotFinished();
};
#endif // CALIBPROCESS_H
