#ifndef MCQMLCAMERAVIEW_H
#define MCQMLCAMERAVIEW_H

#include <QQuickItem>
#include <QTimer>
#include <QSGTexture>
#include <QOpenGLFunctions>
#include <QOpenGLWidget>
#include <QtGui/QOpenGLShaderProgram>
#include <QPointer>
#include "D360Parser.h"

#include "CaptureThread.h"
//#include "ProcessingThread.h"
#include "Structures.h"
#include "D360Stitcher.h"
#include "SharedImageBuffer.h"


QT_FORWARD_DECLARE_CLASS(QGLShaderProgram);

#define PBO_COUNT 2

class OpenCVTexture : public QSGTexture, protected QOpenGLFunctions
{

public:
	OpenCVTexture();
	virtual ~OpenCVTexture();

	inline void bind() {
		glBindTexture(GL_TEXTURE_2D, m_textureId);
	}

	int textureId() const {
		int tex = (int)m_textureId;
		return tex;
	}

	inline QSize textureSize() const {
		return QSize(imageWidth, imageHeight);
	}

	inline bool hasAlphaChannel() const {
		return false;
	}

	inline bool hasMipmaps() const {
		return true;
	}

	inline QSGTexture* texture() const {
		return (QSGTexture*)this;
	}

	void updateTexture(int textureId, int width, int height, QImage::Format format);
	//void updateTexture(byte* data, int width, int height, QImage::Format format);

	bool isValid();
	bool isUpdated();
	void validate();
private:
	//bool m_ownTextureID;
	GLuint m_textureId;
	QSize m_viewportSize;
	bool m_blFirst;

	unsigned int imageWidth;
	unsigned int imageHeight;
	int m_ImageFormatValue;
	bool m_blTextureUpdated;
};


class MCQmlCameraView : public QQuickItem
{
	Q_OBJECT
	Q_PROPERTY(QString cameraViewName READ cameraViewName NOTIFY setCameraViewName)	
	Q_PROPERTY(int cameraNumber READ cameraNumber NOTIFY cameraNumberChanged)

public:
	MCQmlCameraView(QQuickItem* parent = 0);
	virtual ~MCQmlCameraView();
	enum CameraViewType
	{
		CameraViewType_Stitch,
		CameraViewType_Standard,
		CameraViewType_Calibrate,
		CameraViewType_Playback
	} camType;

	QSGNode* updatePaintNode(QSGNode*, UpdatePaintNodeData*);

	virtual void resizeEvent(QResizeEvent * event);

	void setSharedImageBuffer(SharedImageBuffer* sharedImageBuffer) { m_sharedImageBuffer = sharedImageBuffer; }
	
	void init(CameraViewType blLiveView, QOpenGLContext* context);
	void setFullScreenMode(bool isFullScreenView, int fullScreen_width, int fullScreen_height);

	void	onMouseMove(QPoint pos);
	void	onMousePress(QPoint pos);
	void	onMouseRelease(QPoint pos);

    void    enableView(bool enabled);

protected:
	void	mouseMoveEvent(QMouseEvent * event);
	void	mousePressEvent(QMouseEvent * event);
	void	mouseReleaseEvent(QMouseEvent * event);
	void	releaseResources();

signals:	
	void setCameraViewName(QString cameraName);
	void cameraNumberChanged(int camreaNumber);
	void sendClose();

public:	
	QString cameraViewName() const;
	void setCameraName(QString cameraName);
	void setCameraNumber(int deviceNum);
	int cameraNumber() const;
	void closeCameraView(){ emit sendClose(); }
	

protected:
	int m_camDeviceNumber;
	SharedImageBuffer* m_sharedImageBuffer;
	
	GlobalAnimSettings	m_gaSettings;
	QOffscreenSurface* m_surface;
	QOpenGLContext* m_context;
	QOpenGLFunctions_2_0* functions_2_0;

protected slots:
	void handleWindowChanged(QQuickWindow *win);

public slots:
	void sync();
	void cleanup();

private:
	QRectF m_oldGeometry;

	bool m_updateConstantly;
	CameraViewType m_blLiveView;
	bool m_blFullScreenView;
	int m_fullScreenWidth, m_fullScreenHeight;
	OpenCVTexture * m_pVideoTexture;
	bool m_isEnabled;

	bool m_keepAspectRatio;
	double m_aspect;
	bool aspectChanged;

	QQuickWindow* m_quickWindow;

	QString m_strCameraName;
	QString m_strCameraInfo;

	float m_orgYaw;
	float m_orgPitch;
	float m_orgRoll;
	bool m_blMousePressed;
	QPoint	m_pressPt;

	void setAspectRatio(float aspectRatio);
};

#endif // TEXTUREIMAGEITEM_H
