#ifndef PANOLOG_H
#define PANOLOG_H

#include <QFile>
#include <qmutex.h>

enum PANO_LOG_LEVEL {
    DEBUG,
    WARNING,
    INFO,
    CRITICAL
};

class PanoLog : public QObject
{
	Q_OBJECT
public:
    PanoLog(QObject* parent = NULL, QString name = "PanoOneDebug.log");
	virtual ~PanoLog();

    bool initialize(PANO_LOG_LEVEL level = WARNING, QString name = "");
    void enableStdout(bool flag = true) { m_isStdout = flag; }
	void setParent(QObject* parent) { m_parent = parent; }

    void critical(QString msg, bool isNotify = false);
	void debug(QString msg, bool isNotify = false);
	void info(QString msg, bool isNotify = false);
	void warning(QString msg, bool isNotify = false);

private:
    int output(QString msg);

    QObject* m_parent;
    QString m_fileName;
    QFile* m_logFile;
    PANO_LOG_LEVEL m_level;
    bool m_isStdout;

	QMutex fileMutex;

signals:
	void notify(int type, QString title, QString message);
};

#endif // PANOLOG_H
