#include "QmlInteractiveView.h"

#include <QSGOpaqueTextureMaterial>
#include <QSGNode>
#include <QQuickWindow>
#include <QQmlApplicationEngine>
#include <QTime>
#include "QThread.h"

#include <QApplication>
#include "Config.h"
#include <QOpenGLFunctions_3_0>
#include "QmlMainWindow.h"

#define VIDEO_WIDTH		1220.00
#define VIDEO_HEIGHT	1000.00

extern QmlMainWindow* g_mainWindow;
extern QThread* g_mainThread;

OculusTexture::OculusTexture(bool ownTextureID)
{
	m_textureId = -1;

	initializeOpenGLFunctions();

	imageWidth = 0;
	imageHeight = 0;

}

OculusTexture::~OculusTexture() {
}

void OculusTexture::updateTexture(int textureId, int width, int height, QImage::Format format)
{
	imageWidth = width;
	imageHeight = height;
	m_textureId = textureId;
}


QmlInteractiveView::QmlInteractiveView(QQuickItem* parent) : QQuickItem(parent),
m_pOculusTexture(0),
m_sharedImageBuffer(0),
m_keepAspectRatio(true),
aspectChanged(false),
//Added By I
isOculus(false),
isModeChanged(true)
{
	setFlag(ItemHasContents, true);
	m_aspect = VIDEO_WIDTH / VIDEO_HEIGHT;
	m_oculus = NULL;
	m_interact = NULL;
	m_blMousePressed = false;
	connect(this, SIGNAL(windowChanged(QQuickWindow*)), this, SLOT(handleWindowChanged(QQuickWindow*)));
	m_interact = new GLSLPanoramaInteract();
}

QmlInteractiveView::~QmlInteractiveView()
{
	if (m_pOculusTexture){
		delete m_pOculusTexture;
		m_pOculusTexture = NULL;
	}
	if (m_interact)
	{
		delete m_interact;
		m_interact = NULL;
	}
}

void QmlInteractiveView::sync()
{
	if (!m_interact->isInitialized())
	{
		if (m_quickWindow)
			//m_interact->setGL(m_quickWindow->openglContext()->functions(), NULL);
			m_interact->setGL(QOpenGLContext::globalShareContext()->functions(), NULL);
		if (m_sharedImageBuffer)
		{
			GlobalAnimSettings* gasettings = m_sharedImageBuffer->getGlobalAnimSettings();
			m_interact->initialize(gasettings->m_panoXRes, gasettings->m_panoYRes);
			
			m_pOculusTexture = new OculusTexture(true);
			m_pOculusTexture->setFiltering(QSGTexture::Linear);
			m_pOculusTexture->setMipmapFiltering(QSGTexture::None);
		}
	}
}

void QmlInteractiveView::cleanup()
{
	if (m_pOculusTexture)
	{
		delete m_pOculusTexture;
		m_pOculusTexture = NULL;
	}
}

void QmlInteractiveView::releaseResources()
{
}

QSGNode* QmlInteractiveView::updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*)
{
	QSGGeometryNode* node = 0;
	QSGGeometry* geometry = 0;

	QSGNode::DirtyState dirtyState;

	if (!m_pOculusTexture)
		return oldNode;
	if (!m_sharedImageBuffer)
		return oldNode;
	if (!m_sharedImageBuffer->getGlobalAnimSettings())
		return oldNode;

	GlobalAnimSettings* gasettings = m_sharedImageBuffer->getGlobalAnimSettings();
	setAspectRatio( 1.0f * gasettings->getPanoXres() / gasettings->getPanoYres());
	

	int textureID = -1;
	m_sharedImageBuffer->lockOculus();
	bool isOculusOn = m_sharedImageBuffer->getGlobalAnimSettings()->m_oculus;
	if (isOculusOn && isOculusCreated())
	{
		textureID = m_oculus->getTextureIDForQML();
	}
	else if (!isOculusOn)
	{
		textureID = m_interact->getTargetGPUResource();
		if (m_sharedImageBuffer->getStitcher() && !m_sharedImageBuffer->getStitcher()->isFinished())
			m_interact->render(m_sharedImageBuffer->getStitcher()->getPanoramaTextureId(), gasettings->isStereo(), gasettings->m_fInteractYaw, gasettings->m_fInteractPitch, gasettings->m_fInteractRoll);
	}
	m_sharedImageBuffer->unlockOculus();
	if (textureID == -1)
		return oldNode;
	m_pOculusTexture->updateTexture(textureID, gasettings->getPanoXres(), gasettings->getPanoYres(), QImage::Format_RGB888);
	
	if (!oldNode) {
		// Create the node.
		node = new QSGGeometryNode;
		geometry = new QSGGeometry(QSGGeometry::defaultAttributes_TexturedPoint2D(), 4);
		geometry->setDrawingMode(GL_TRIANGLE_STRIP);
		node->setGeometry(geometry);
		node->setFlag(QSGNode::OwnsGeometry);
		
		QSGOpaqueTextureMaterial* material = new QSGOpaqueTextureMaterial; //QSGTextureMaterial  can be used for opacity support
		material->setTexture(m_pOculusTexture);
		node->setMaterial(material);
		node->setFlag(QSGNode::OwnsMaterial);
	}
	else {
		node = static_cast<QSGGeometryNode *>(oldNode);
		geometry = node->geometry();
	}

	if (m_oldGeometry != boundingRect() || aspectChanged || isModeChanged)
	{
		aspectChanged = false;
		isModeChanged = false;
		// Create the vertices and map to texture.
		makeVtTexForRect(boundingRect(), m_aspect, geometry->vertexDataAsTexturedPoint2D(), m_keepAspectRatio, isOculusOn);
		dirtyState |= QSGNode::DirtyGeometry;

		update();
	}
	node->markDirty(dirtyState);
	m_oldGeometry = boundingRect();
	return node;
}

void QmlInteractiveView::setAspectRatio(float aspectRatio)
{
	if (m_aspect != aspectRatio)
	{
		m_aspect = aspectRatio;
		aspectChanged = true;
	}
}

bool QmlInteractiveView::isOculusCreated()
{
	if (!m_oculus) return false;
	return m_oculus->isCreated();
}

void QmlInteractiveView::mouseMoveEvent(QMouseEvent * event)
{
}

void QmlInteractiveView::mousePressEvent(QMouseEvent * event)
{

}

void QmlInteractiveView::mouseReleaseEvent(QMouseEvent * event)
{
}

void QmlInteractiveView::onMouseMove(QPoint pos)
{
	if (!m_sharedImageBuffer)
		return;
	//if (abs(m_pressPt.x() - pos.x()) < 4 || abs(m_pressPt.y() - pos.y()) < 4)
	//	return;
	GlobalAnimSettings& setting = g_mainWindow->getGlobalAnimSetting();
	if (m_blMousePressed && m_crossPt != vec3_null)
	{
		vec3 crossPt = getCrossPt(pos);
		if (crossPt != vec3_null)
		{
			mat3 m = mat3_id, mOrg = mat3_id, mOffset = mat3_id, invM = mat3_id;
			vec3 u(setting.m_fInteractRoll * sd_to_rad, setting.m_fInteractPitch * sd_to_rad, setting.m_fInteractYaw * sd_to_rad);
			mOrg.set_rot_zxy(u);

			sd_scalar rotAngle = 0;
			vec3 rotAxis;
			get_rot(vec3_z, m_crossPt, crossPt, rotAngle, rotAxis);

			mOffset.set_rot(rotAngle, rotAxis);

			m = mult(mOffset, mOrg);
			m.get_rot_zxy(u);
			setting.m_fInteractRoll = 0; // u[0] * sd_to_deg;
			setting.m_fInteractPitch = u[1] * sd_to_deg;
			setting.m_fInteractYaw = u[2] * sd_to_deg;

			m_crossPt = crossPt;
		}
	}

}

void QmlInteractiveView::onMousePress(QPoint pos)
{
	if (!m_sharedImageBuffer)
		return;
	GlobalAnimSettings& setting = g_mainWindow->getGlobalAnimSetting();
	m_blMousePressed = true;
	m_pressPt = pos;

	//m_orgYaw = setting.m_fInteractYaw;
	//m_orgPitch = setting.m_fInteractPitch;
	//m_orgRoll = setting.m_fInteractRoll;


	m_crossPt = getCrossPt(pos);
}

vec3 QmlInteractiveView::getCrossPt(QPoint pos)
{
	QRectF bounds = boundingRect();
	int iWidth = bounds.width();
	int iHeight = bounds.height();
	int iPosX = pos.x();
	int iPosY = pos.y();

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
	int left, top;
	left = (iWidth - viewWidth) / 2;
	top = (iHeight - viewHeight) / 2;
	mat4 mProj = m_interact->getProjMat();
	mat4 mView = m_interact->getViewMat();
	vec3 pickRayDir = vec3_neg_z, pickOrg = vec3_null;
	getPickRay(iPosX, iPosY, mat4_id, mProj, left, top, viewWidth, viewHeight, pickRayDir, pickOrg);
	pickRayDir.normalize();
	sd_scalar t = intersects(pickOrg, pickRayDir, vec3_null, 1, false);
	if (t == FLT_MAX)
	{
		return vec3_null;
	}
	vec3 crossPt = pickOrg + pickRayDir * t;
	return crossPt;
}

void QmlInteractiveView::onMouseRelease(QPoint pos)
{
	if (!m_sharedImageBuffer)
		return;
	m_blMousePressed = false;
	m_crossPt = vec3_null;
}

void QmlInteractiveView::renderStarted()
{
	//Added By I
// 	if (isOculus && m_oculus && m_oculus->isCreated())
// 	{
// 		glEnable(GL_FRAMEBUFFER_SRGB);
// 	}
// 	else
// 		glDisable(GL_FRAMEBUFFER_SRGB);
}

void QmlInteractiveView::renderFinished()
{
	//Added By I
	/*glDisable(GL_FRAMEBUFFER_SRGB);*/
}

void QmlInteractiveView::setSharedImageBuffer(SharedImageBuffer* sharedImageBuffer)
{
	m_sharedImageBuffer = sharedImageBuffer;
	//moveToThread(m_quickWindow->thread());
	//m_quickWindow->openglContext()->makeCurrent(m_quickWindow->openglContext()->surface());
	//m_quickWindow->openglContext()->doneCurrent();
	//moveToThread(g_mainThread);
}

void QmlInteractiveView::handleWindowChanged(QQuickWindow *win)
{
	if (m_quickWindow != win)
	{
		if (win) {
			//connect(win, SIGNAL(beforeSynchronizing()), this, SLOT(sync()), Qt::DirectConnection);
			//connect(win, SIGNAL(sceneGraphInvalidated()), this, SLOT(cleanup()), Qt::DirectConnection);
			connect(win, SIGNAL(beforeRendering()), this, SLOT(renderStarted()), Qt::DirectConnection);
			connect(win, SIGNAL(afterRendering()), this, SLOT(renderFinished()), Qt::DirectConnection);
			connect(win, SIGNAL(afterSynchronizing()), this, SLOT(update()));
			connect(win, SIGNAL(beforeSynchronizing()), this, SLOT(sync()), Qt::DirectConnection);
			connect(win, SIGNAL(sceneGraphInvalidated()), this, SLOT(cleanup()), Qt::DirectConnection);
		}
		m_quickWindow = win;
	}
}