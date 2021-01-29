#pragma once
#include <QObject>
#include "CoreEngine.h"

class ConsoleRunner : public QObject
{
	Q_OBJECT

public:
	ConsoleRunner();
	virtual ~ConsoleRunner();

	bool initPaths(QString path, QString url);
	void run();

private:
	QString m_iniPath;
	QString m_streamPath;

	void openIniPath( QString iniPath );
	void openProject( QString iniPath );
	void startStream( QString streamPath, int width, int height, int streamMode );
public slots:
	void startStream(bool bFinish);
// signals:
// 	void started( bool isStarted );

};

