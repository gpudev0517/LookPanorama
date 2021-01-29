#ifndef QMLAPPLICATIONSETTING_H
#define QMLAPPLICATIONSETTING_H

#include <QQuickItem>
#include "QmlMainWindow.h"
#include <QApplication>
#include "SharedImageBuffer.h"
#include "D360Parser.h"
#include "define.h"
#include "SlotInfo.h"
#include <QDir>
#include <QtXml/QDomDocument>
#include <QQuickWindow>
#include <iostream>
#include <sstream>
#include <QFile>
#include <QScreen>
#include <QDesktopWidget>
#include <QtMath>
#include "3DMath.h" 

class QmlApplicationSetting : public QObject {

	Q_OBJECT
	Q_PROPERTY(QString theme READ theme WRITE setTheme NOTIFY themeChanged)
	Q_PROPERTY(bool useCUDA READ useCUDA WRITE setUseCUDA NOTIFY useCUDAChanged)
	Q_PROPERTY(bool isCudaAvailable READ isCudaAvailable NOTIFY cudaAvailableChanged)
	Q_PROPERTY(QString sessionTkgCapturePath READ sessionTkgCapturePath WRITE setSessionTkgCapturePath NOTIFY sessionTkgCapturePathChanged)
	Q_PROPERTY(bool streamingMode READ streamingMode WRITE setStreamingMode NOTIFY streamingModeChanged)
public:
	QmlApplicationSetting(QObject* parent = 0);
	virtual ~QmlApplicationSetting();

	bool useCUDA() const;
	bool isCudaAvailable() const;
	QString theme() const;
	QString sessionTkgCapturePath() const;
	bool streamingMode() const;

signals:
	void useCUDAChanged(const bool useCUDA);
	void themeChanged(const QString useTheme);

	void cudaAvailableChanged(bool cudaAvailable);
	void sessionTkgCapturePathChanged(QString sessionTkgCapturePath);
	void streamingModeChanged(bool streamingMode);


public slots:
	void setUseCUDA(bool useCUDA);
	void loadApplicationSetting();
	void saveApplicationSetting();

	void setTheme(QString theme);
	void setSessionTkgCapturePath(QString sessioinTkgCapturePath);

	void setStreamingMode(bool streamingMode);

protected:

private:
	bool m_useCUDA;		
	bool m_isCudaAvailable;
	QString m_theme;
	QString m_sessionTkgCapturePath;
	bool m_streamingMode;
};


#endif // QMLAPPLICATIONSETTING_H