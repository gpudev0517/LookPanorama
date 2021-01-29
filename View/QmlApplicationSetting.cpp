#include "QmlApplicationSetting.h"
#include "CoreEngine.h"

//extern CoreEngine* g_Engine;
//extern bool g_useCUDA;

QmlApplicationSetting::QmlApplicationSetting(QObject* parent) : QObject(parent)
{
	m_sessionTkgCapturePath = "C:\CapturePath";
	
	
#ifdef USE_CUDA
	int devCount = -1;
	cudaGetDeviceCount(&devCount);
	if (devCount > 0){
		m_isCudaAvailable = true;
		setUseCUDA(true);
	}
	else{
		m_isCudaAvailable = false;
		setUseCUDA( false );
	}
#else
	m_isCudaAvailable = false;
	setUseCUDA( false );
#endif
	loadApplicationSetting();

	if (!m_isCudaAvailable)
		setUseCUDA( false );

	emit cudaAvailableChanged(m_isCudaAvailable);
	emit useCUDAChanged(m_useCUDA);
}

QmlApplicationSetting::~QmlApplicationSetting()
{
	saveApplicationSetting();
}

bool QmlApplicationSetting::useCUDA() const
{
	return m_useCUDA;
}

bool QmlApplicationSetting::isCudaAvailable() const
{
	return m_isCudaAvailable;
}

void QmlApplicationSetting::setUseCUDA(bool useCUDA)
{
	m_useCUDA = useCUDA;	
	g_useCUDA = m_useCUDA;
	emit useCUDAChanged(useCUDA);
}

void QmlApplicationSetting::loadApplicationSetting() 
{
	QSettings settings("Centroid LAB", "Look3D");

	settings.beginGroup("appSetting");
	setUseCUDA(settings.value("useCUDA", false).toBool());
	setTheme(settings.value("theme").toString());
	setSessionTkgCapturePath(settings.value("SessionTakeCapturePath").toString());
	setStreamingMode(settings.value("StreamingMode").toBool());
	settings.endGroup();
}

void QmlApplicationSetting::saveApplicationSetting() {

	QSettings settings("Centroid LAB", "Look3D");

	settings.beginGroup("appSetting");
	settings.setValue("useCUDA", m_useCUDA);
	settings.setValue("theme", m_theme);
	settings.setValue("SessionTakeCapturePath", m_sessionTkgCapturePath);
	settings.setValue("StreamingMode", m_streamingMode);
	settings.endGroup();
}

QString QmlApplicationSetting::theme() const
{
	return m_theme;
}

void QmlApplicationSetting::setTheme(QString theme)
{
	if (m_theme == theme)
			return;
	
		m_theme = theme;
		emit themeChanged(theme);
}

QString QmlApplicationSetting::sessionTkgCapturePath() const{
	return m_sessionTkgCapturePath;
}

bool QmlApplicationSetting::streamingMode() const
{
	return m_streamingMode;
}

void QmlApplicationSetting::setSessionTkgCapturePath(QString sessionTkgCapturePath)
{
	if (sessionTkgCapturePath == m_sessionTkgCapturePath)
		return;

	m_sessionTkgCapturePath = sessionTkgCapturePath;
	emit sessionTkgCapturePathChanged(sessionTkgCapturePath);
}

void QmlApplicationSetting::setStreamingMode(bool streamingMode)
{
	if (streamingMode == m_streamingMode)
		return;

	m_streamingMode = streamingMode;
	saveApplicationSetting();
	emit streamingModeChanged(streamingMode);
}
