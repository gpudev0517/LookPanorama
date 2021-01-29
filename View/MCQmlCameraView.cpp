#include "MCQmlCameraView.h"

#include <QSGOpaqueTextureMaterial>
#include <QSGNode>
#include <QQuickWindow>
#include <QQmlApplicationEngine>
#include <QTime>
#include "QThread.h"
#include "3DMath.h"
#include "QmlMainWindow.h"

extern QmlMainWindow* g_mainWindow;

QMutex g_getPanoramaMutex;

#define VIDEO_WIDTH		1220.00
#define VIDEO_HEIGHT	1000.00

OpenCVTexture::OpenCVTexture()
{
	m_textureId = -1;
	m_blFirst = true;

	initializeOpenGLFunctions();

	m_blTextureUpdated = false;
	imageWidth = 0;
	imageHeight = 0;
}


OpenCVTexture::~OpenCVTexture()
{
}

void OpenCVTexture::updateTexture(int textureId, int width, int height, QImage::Format format)
{
	if (m_textureId != textureId)
	{
		imageWidth = width;
		imageHeight = height;
		m_textureId = textureId;
		m_blTextureUpdated = true;
	}
}

bool OpenCVTexture::isValid()
{
	return m_textureId != -1;
}

bool OpenCVTexture::isUpdated()
{
	return m_blTextureUpdated;
}

void OpenCVTexture::validate()
{
	m_blTextureUpdated = false;
}

//void OpenCVTexture::updateTexture(byte* data, int width, int height, QImage::Format format)
//{
//	glBindTexture(GL_TEXTURE_2D, m_textureId);
//	if (imageWidth != width || imageHeight != height)
//	{
//		imageWidth = width;
//		imageHeight = height;
//		if (format == QImage::Format_Indexed8) {
//			glTexImage2D(GL_TEXTURE_2D,     // Type of texture
//				0,                 // Pyramid level (for mip-mapping) - 0 is the top level
//				GL_DEPTH_COMPONENT,            // Internal colour format to convert to
//				width,          // Image width
//				height,          // Image height
//				0,                 // Border width in pixels (can either be 1 or 0)
//				GL_DEPTH_COMPONENT, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
//				GL_UNSIGNED_BYTE,  // Image data type
//				NULL);        // The actual image data itself
//		}
//		else if (format == QImage::Format_RGB888){
//			glTexImage2D(GL_TEXTURE_2D,     // Type of texture
//				0,                 // Pyramid level (for mip-mapping) - 0 is the top level
//				GL_RGB,            // Internal colour format to convert to
//				width,          // Image width
//				height,          // Image height
//				0,                 // Border width in pixels (can either be 1 or 0)
//				GL_RGB, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
//				GL_UNSIGNED_BYTE,  // Image data type
//				NULL);        // The actual image data itself
//		}
//		else if (format == QImage::Format_RGB32){
//			glTexImage2D(GL_TEXTURE_2D,     // Type of texture
//				0,                 // Pyramid level (for mip-mapping) - 0 is the top level
//				GL_RGBA,            // Internal colour format to convert to
//				width,          // Image width
//				height,          // Image height
//				0,                 // Border width in pixels (can either be 1 or 0)
//				GL_RGBA, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
//				GL_UNSIGNED_BYTE,  // Image data type
//				NULL);        // The actual image data itself
//		}
//	}
//	if (data != nullptr)
//	{
//		if (format == QImage::Format_Indexed8)
//		{
//			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, data);
//		}
//		else if (format == QImage::Format_RGB888)
//		{
//			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
//		}
//		else if (format == QImage::Format_RGB32)
//		{
//			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
//		}
//	}
//}




MCQmlCameraView::MCQmlCameraView(QQuickItem* parent) : QQuickItem(parent),
m_pVideoTexture(0),
m_keepAspectRatio(true),
aspectChanged(false),
m_sharedImageBuffer(NULL),
m_isEnabled(true)
{
	setFlag(ItemHasContents, true);
	m_aspect = VIDEO_WIDTH / VIDEO_HEIGHT;
	m_camDeviceNumber = 0;
	m_blLiveView = CameraViewType_Standard;
	m_blFullScreenView = false;
	m_strCameraName = "";
	m_strCameraInfo = "";
	connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));

	setAcceptedMouseButtons(Qt::MouseButtonMask);

	m_blMousePressed = false;
	m_orgYaw = m_orgPitch = m_orgRoll = 0;
}

void MCQmlCameraView::resizeEvent(QResizeEvent *event) 
{
}

MCQmlCameraView::~MCQmlCameraView()
{
	if (m_pVideoTexture){
		delete m_pVideoTexture;
		m_pVideoTexture = NULL;
	}
}

void MCQmlCameraView::handleWindowChanged(QQuickWindow *win)
{
	if (m_quickWindow != win)
	{
		if (win) {
			connect(win, SIGNAL(beforeSynchronizing()), this, SLOT(sync()), Qt::DirectConnection);
			connect(win, SIGNAL(sceneGraphInvalidated()), this, SLOT(cleanup()), Qt::DirectConnection);
			connect(win, SIGNAL(afterSynchronizing()), this, SLOT(update()));
			//! [1]
			// If we allow QML to do the clearing, they would clear what we paint
			// and nothing would show.
			//! [3]
			win->setClearBeforeRendering(false);
		}
		m_quickWindow = win;
	}
}

void MCQmlCameraView::sync()
{
	if (!m_pVideoTexture)
	{
		m_pVideoTexture = new OpenCVTexture();
		m_pVideoTexture->setFiltering(QSGTexture::Linear);
		m_pVideoTexture->setMipmapFiltering(QSGTexture::None);
		m_pVideoTexture->setHorizontalWrapMode(QSGTexture::WrapMode::ClampToEdge);
		m_pVideoTexture->setVerticalWrapMode(QSGTexture::WrapMode::ClampToEdge);
	}
}

void MCQmlCameraView::cleanup()
{
	if (m_pVideoTexture) {
		delete m_pVideoTexture;
		m_pVideoTexture = NULL;
	}
}

void MCQmlCameraView::releaseResources()
{
}

QString MCQmlCameraView::cameraViewName() const
{
	return m_strCameraName + m_strCameraInfo;
}

void MCQmlCameraView::setCameraName(QString cameraName)
{
	m_strCameraName = cameraName;
	emit setCameraViewName(m_strCameraName + m_strCameraInfo);
}

void MCQmlCameraView::setCameraNumber(int deviceNum)
{
	m_camDeviceNumber = deviceNum;
	emit cameraNumberChanged(m_camDeviceNumber);
}

int MCQmlCameraView::cameraNumber() const
{
	return m_camDeviceNumber;
}

QSGNode* MCQmlCameraView::updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*)
{
	m_strCameraInfo = "";
	if (!m_isEnabled)
		return oldNode;
	if (!m_sharedImageBuffer || !m_sharedImageBuffer->getGlobalAnimSettings())
		return oldNode;

	QSGGeometryNode* node = 0;
	QSGGeometry* geometry = 0;

	QSGNode::DirtyState dirtyState;
	GlobalAnimSettings* gasettings = m_sharedImageBuffer->getGlobalAnimSettings();
	D360Stitcher* stitcher = m_sharedImageBuffer->getStitcher().get();

	float aspect = m_aspect;

	if (m_blLiveView == CameraViewType_Standard)
	{
		int width = gasettings->m_xres;
		int height = gasettings->m_yres;
		m_strCameraInfo = "(" + QString::number(width) + "x" + QString::number(height) + ", " + QString::number(gasettings->getRealFps(), 'f', 1) + "fps)";
		emit setCameraViewName(m_strCameraName + m_strCameraInfo);
		if (gasettings->cameraSettingsList().size() > m_camDeviceNumber && m_camDeviceNumber >= 0)
		{
			width = gasettings->cameraSettingsList()[m_camDeviceNumber].xres;
			height = gasettings->cameraSettingsList()[m_camDeviceNumber].yres;
		}
		if (width != 0)
		{
			aspect = ((double)width) / height;
			if (stitcher)
			{
				int rawTextureID = stitcher->getRawTextureId(m_camDeviceNumber);
				m_pVideoTexture->updateTexture(rawTextureID, width, height, QImage::Format_RGB888);
			}
		}
	}
	else if (m_blLiveView == CameraViewType_Stitch)
	{
		int panoXRes = gasettings->getPanoXres();
		int panoYRes = gasettings->getPanoYres();
		if (gasettings->isStereo())
			panoYRes *= 2;
		aspect = ((double)panoXRes) / panoYRes;
		if (m_blFullScreenView) {
			aspect = ((double)m_fullScreenWidth) / m_fullScreenHeight;
		}

		m_pVideoTexture->updateTexture(stitcher->getPanoramaTextureId(), panoXRes, panoYRes, QImage::Format_RGB888);
	}
	else if (m_blLiveView == CameraViewType_Calibrate)
	{
		int width = gasettings->m_xres;
		int height = gasettings->m_yres;
		if (gasettings->cameraSettingsList().size() > m_camDeviceNumber && m_camDeviceNumber >= 0)
		{
			width = gasettings->cameraSettingsList()[m_camDeviceNumber].xres;
			height = gasettings->cameraSettingsList()[m_camDeviceNumber].yres;
		}
		if (width != 0)
		{
			aspect = ((double)width) / height;
			if (stitcher && stitcher->viewProcessers())
			{
				int rawTextureID = stitcher->getRawTextureId(m_camDeviceNumber);
				m_pVideoTexture->updateTexture(rawTextureID, width, height, QImage::Format_RGB888);
			}
		}
	}
	else if (m_blLiveView == CameraViewType_Playback)
	{
		int panoXRes = gasettings->getPanoXres();
		int panoYRes = gasettings->getPanoYres();
		if (gasettings->isStereo())
			panoYRes *= 2;
		aspect = ((double)panoXRes) / panoYRes;

		PlaybackStitcher* playbackStitcher = m_sharedImageBuffer->getPlaybackStitcher().get();
		if (playbackStitcher)
		{
			int rawTextureID = playbackStitcher->getPanoramaTextureId();
			m_pVideoTexture->updateTexture(rawTextureID, panoXRes, panoYRes, QImage::Format_RGB888);
		}
	}

	setAspectRatio(aspect);
	if (!m_pVideoTexture->isValid())
		return NULL;

	if (!oldNode) {
		// Create the node.
		node = new QSGGeometryNode;
		geometry = new QSGGeometry(QSGGeometry::defaultAttributes_TexturedPoint2D(), 4);
		geometry->setDrawingMode(GL_TRIANGLE_STRIP);
		node->setGeometry(geometry);
		node->setFlag(QSGNode::OwnsGeometry);

		QSGOpaqueTextureMaterial* material = new QSGOpaqueTextureMaterial; //QSGTextureMaterial  can be used for opacity support
		material->setTexture(m_pVideoTexture);

		node->setMaterial(material);
		node->setFlag(QSGNode::OwnsMaterial);
	}
	else
	{
		node = static_cast<QSGGeometryNode *>(oldNode);
		geometry = node->geometry();

		if (m_pVideoTexture->isUpdated()) {
			((QSGOpaqueTextureMaterial*)node->material())->setTexture(m_pVideoTexture);
			m_pVideoTexture->validate();
		}
	}

	if (m_oldGeometry != boundingRect() || aspectChanged)
	{
		aspectChanged = false;
		// Create the vertices and map to texture.
		makeVtTexForRect(boundingRect(), m_aspect, geometry->vertexDataAsTexturedPoint2D(), m_keepAspectRatio, (m_blLiveView == CameraViewType_Stitch || m_blLiveView == CameraViewType_Playback) ? true : false);
		dirtyState |= QSGNode::DirtyGeometry;
		update();
	}
	node->markDirty(dirtyState);
	m_oldGeometry = boundingRect();
	return node;
}

void MCQmlCameraView::setAspectRatio(float aspectRatio)
{
	if (m_aspect != aspectRatio)
	{
		m_aspect = aspectRatio;
		aspectChanged = true;
	}
}

void MCQmlCameraView::init(CameraViewType blLiveView, QOpenGLContext* context)
{
	m_blLiveView = blLiveView;
	m_oldGeometry = QRectF();
}

void MCQmlCameraView::setFullScreenMode(bool isFullScreenView, int fullScreen_width, int fullScreen_height)
{	
	m_blFullScreenView = isFullScreenView;
	m_fullScreenWidth = fullScreen_width;
	m_fullScreenHeight = fullScreen_height;
}

void MCQmlCameraView::enableView(bool enabled)
{
	m_isEnabled = enabled;
}

void MCQmlCameraView::mouseMoveEvent(QMouseEvent * event)
{
}

void MCQmlCameraView::mousePressEvent(QMouseEvent * event)
{
}

void MCQmlCameraView::mouseReleaseEvent(QMouseEvent * event)
{
}

void MCQmlCameraView::onMouseMove(QPoint pos)
{
	if (!m_sharedImageBuffer)
		return;
	GlobalAnimSettings& setting = g_mainWindow->getGlobalAnimSetting();

	QRectF bounds = boundingRect();
	int iWidth = bounds.width();
	int iHeight = bounds.height();
	int iPosX = pos.x();
	int iPosY = pos.y();
	if (m_blMousePressed)
	{
		bool xKeyPressed = (GetAsyncKeyState(Qt::Key_X) ? true : false);
		bool yKeyPressed = (GetAsyncKeyState(Qt::Key_Y) ? true : false);

		float fPanoAspect = m_aspect;
		float fViewAspect = iWidth / (float)iHeight;
		int viewWidth, viewHeight;
		if (fPanoAspect > fViewAspect)
		{
			viewWidth = iWidth;
			viewHeight = iWidth / fPanoAspect;
		}
		else
		{
			viewHeight = iHeight;
			viewWidth = iHeight * fPanoAspect;
		}
		int cPx = iWidth / 2;
		int cPy = iHeight / 2;
		float scaleYaw, scalePitch, scaleRoll;
		scaleYaw = sd_two_pi * sd_to_deg / viewWidth;
		scalePitch = sd_pi * sd_to_deg / viewHeight;
		scaleRoll = sd_two_pi * sd_to_deg / viewHeight;
		float wYaw, wPitch, wRoll;
		// view -x:	-			-			    cP				-				-
		// Sphere:  back		left			front			right			back
		// wRoll:	0			1				0				-1				0
		// wPitch:	-1			0				1				0				-1
		// wYaw:	1			1				1				1				1
		wYaw = 1;
		wPitch = 2 * (1 - sd_min(1, 2 * abs(cPx - m_pressPt.x()) / (float)viewWidth)) - 1;
		wRoll = 1 - sd_min(abs(cPx + iWidth / 4 - m_pressPt.x()), abs(cPx - iWidth / 4 - m_pressPt.x())) * 4 / (float)viewWidth;
		wRoll = m_pressPt.x() > cPx ? wRoll : -wRoll;

		float yaw = wYaw * scaleYaw * (iPosX - m_pressPt.x());
		float pitch = wPitch * scalePitch * (iPosY - m_pressPt.y());
		float roll = wRoll * scaleRoll * (iPosY - m_pressPt.y());
#if 1
		mat3 m = mat3_id, mOrg = mat3_id, mOffset = mat3_id, invM = mat3_id;
		vec3 u(m_orgRoll * sd_to_rad, m_orgPitch * sd_to_rad, m_orgYaw * sd_to_rad);
		mOrg.set_rot_zxy(u);
		u[0] = roll * sd_to_rad; u[1] = pitch * sd_to_rad, u[2] = yaw * sd_to_rad;
		mOffset.set_rot_zxy(u);
		m = mult(mOffset, mOrg);
		m.get_rot_zxy(u);

		if (xKeyPressed) {
			setting.m_fYaw = u[2] * sd_to_deg;
		} else if (yKeyPressed) {
			setting.m_fPitch = u[1] * sd_to_deg;
		}
		else {
			setting.m_fYaw = u[2] * sd_to_deg;
			setting.m_fPitch = u[1] * sd_to_deg;
			setting.m_fRoll = u[0] * sd_to_deg;
		}		
		
#else
		setting.m_fYaw = m_orgYaw + yaw;
		setting.m_fPitch = m_orgPitch + pitch;
		setting.m_fRoll = m_orgRoll + roll;
#endif
		m_sharedImageBuffer->getStitcher()->restitch();
	}
}

void MCQmlCameraView::onMousePress(QPoint pos)
{
	if (!m_sharedImageBuffer)
		return;
	GlobalAnimSettings& setting = g_mainWindow->getGlobalAnimSetting();
	m_blMousePressed = true;
	m_pressPt = pos;
	m_orgYaw = setting.m_fYaw;
	m_orgPitch = setting.m_fPitch;
	m_orgRoll = setting.m_fRoll;
}

void MCQmlCameraView::onMouseRelease(QPoint pos)
{
	if (!m_sharedImageBuffer)
		return;
	m_blMousePressed = false;
}
