#include "OculusViewer.h"
#include <QTime>
#include <QApplication>

#include "Config.h"
#include "common.h"
//#include "define.h"
#include <QOpenGLFunctions_4_3_Core>
#include "GPUProgram.h"

#define HORZ_SEGS 256
#define VERT_SEGS 128

#if DEBUG_OCULUS_VIEWER
#define OCULUS_VIEWER_GRID_CORNERS 10
#endif


// Oculus Device
OculusRender::OculusRender(QOpenGLContext *context)
//: QOpenGLWidget(parent)
//: QObject(parent)
: mirrorTexture(nullptr)
, renderTexture(nullptr)
, depthBuffer(nullptr)
, vertices(0)
, texCoords(0)
, texCoordsRight(0)
, verticesForQML(0)
, texCoordsForQML(0)
, oculusThreadInstance(0)
, m_program(0)
#if DEBUG_OCULUS_VIEWER
, m_gridProgram(0)
, verticesGrid(0)
#endif
, m_isCreated(false)
, m_finished(true)
, isDisconnecting(false)
, m_Name("OculusRender")
#ifdef OCULUS_1_8_0
, frameIndex(0)
#endif
, m_context(0)
, mirrorFBO(-1)
, mirrorFBOForQML(-1)
, mirrorTextureForQML(-1)
, isChessboard(false)
{
	// 1. Initialize LibOVR through ovr_Initialize.
	PANO_LOG("Initializing Oculus SDK");
	ovrResult result = ovr_Initialize(NULL);
	if (OVR_FAILURE(result)) {
		ovrErrorInfo errorInfo;
		ovr_GetLastErrorInfo(&errorInfo);
		PANO_N_ERROR(QString("ovr_Initialize failed: %1").arg(errorInfo.ErrorString));
		return;
	}

	// 2. Call ovr_Create and check the return value to see if it succeeded.You can periodically poll for the
	//	presence of an HMD with ovr_GetHmdDesc(nullptr).
	ovrGraphicsLuid luid;
	result = ovr_Create(&session, &luid);
	if (OVR_FAILURE(result))
	{
		PANO_N_ERROR("HMD creation failed.");
//#ifdef OCULUS_1_8_0
		ovr_Shutdown();
//#endif
		return;
	}

	hmdDesc = ovr_GetHmdDesc(session);
	PANO_LOG(QString("HMD created: %1").arg(hmdDesc.ProductName));


	ovrSizei windowSize = hmdDesc.Resolution;
	resize(windowSize.w, windowSize.h);

	//setAutoSwapBuffer(false);
	//and call swapBuffer()

	setSurfaceType(QSurface::OpenGLSurface);
	QSurfaceFormat format;
	format.setDepthBufferSize(16);
	format.setStencilBufferSize(8);
	// Oculus render should be in with OpenGL4.3 or above.
	//format.setVersion(4, 3);
	format.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);

	format.setSwapInterval(0);
	format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
	setFormat(format);

	m_context = new QOpenGLContext;
	m_context->setFormat(format);
	m_context->setShareContext(QOpenGLContext::globalShareContext());
	if (!m_context->create()) {
		PANO_N_LOG("OpenGL context creating is failed!");
		return;
	}

	this->setGeometry(windowSize.w, windowSize.h, 0, 0);
	setTitle("OculusRift Viewer");	
	show();
 	this->hide();

	m_isCreated = true;

	//
	// Initialize variables(s)
	//
	doStop = false;
}

OculusRender::~OculusRender()
{
	disconnect();
	//m_context->moveToThread(QApplication::instance()->thread());
	//if (oculusThreadInstance) delete oculusThreadInstance;
	
//#ifndef OCULUS_1_8_0
	if (m_isCreated)
	{
		ovr_Destroy(session);
	}
//#endif
	ovr_Shutdown();

	

	if (mirrorFBO != -1)
	{
		m_context->functions()->glDeleteFramebuffers(1, &mirrorFBO);
	}

	if (mirrorFBOForQML != -1)
	{
		m_context->functions()->glDeleteTextures(1, &mirrorTextureForQML);
		m_context->functions()->glDeleteFramebuffers(1, &mirrorFBOForQML);
	}

	if (m_context)
	{
		m_context->doneCurrent();
		delete m_context;
		m_context = 0;
	}

	if (m_program)
	{
		delete m_program;
		m_program = 0;
	}

#if DEBUG_OCULUS_VIEWER
	if (m_gridProgram)
	{
		delete m_gridProgram;
		m_gridProgram = 0;
	}
#endif
	this->sharedImageBuffer = NULL;
	m_isCreated = false;
}

bool OculusRender::close()
{
	QApplication::instance()->quit();
	return true;
}

void OculusRender::initialize(SharedImageBuffer *sharedImageBuffer, int width, int height)
{
	this->sharedImageBuffer = sharedImageBuffer;

	m_panoramaWidth = width;
	m_panoramaHeight = height;

	initializeGL(sharedImageBuffer->getGlobalAnimSettings()->isStereo());
}

void
OculusRender::setPanoramaTexture(int panoramaTexture)
{
	m_panoramaTexture = panoramaTexture;
}

void
OculusRender::initializeGL(bool isStereo)
{
	if (!m_isCreated) return;

	this->isStereo = isStereo;

	m_context->makeCurrent(this);

	m_context->functions()->initializeOpenGLFunctions();

	// 4. Initialize rendering for the HMD.
	// a.Select rendering parameters such as resolution and field of view based on HMD capabilities.
	// ?See : ovr_GetFovTextureSize and ovr_GetRenderDesc.
	// b.Configure rendering by creating D3D / OpenGL - specific swap texture sets to present data to the headset.
	// ?See : ovr_CreateSwapTextureSetD3D11 andovr_CreateSwapTextureSetGL.

	// Configure Stereo settings.
	ovrSizei recommenedTex0Size = ovr_GetFovTextureSize(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0], 1);
	ovrSizei recommenedTex1Size = ovr_GetFovTextureSize(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1], 1);
	ovrSizei bufferSize;
	bufferSize.w = recommenedTex0Size.w + recommenedTex1Size.w;
	bufferSize.h = recommenedTex0Size.h > recommenedTex1Size.h ? recommenedTex0Size.h : recommenedTex1Size.h;

#ifdef OCULUS_1_8_0
	renderTexture = new TextureBuffer(session, true, true, bufferSize, 1, NULL, 1, m_context->functions());
	depthBuffer = new DepthBuffer(renderTexture->GetSize(), 0);
	if (!renderTexture->TextureChain)
	{
		T_WARNING("Failed to create Oculus texture.");
		return;
	}

	ovrSizei windowSize = { hmdDesc.Resolution.w, hmdDesc.Resolution.h };

	ovrMirrorTextureDesc desc;
	memset(&desc, 0, sizeof(desc));
	desc.Width = windowSize.w;
	desc.Height = windowSize.h;
	desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;

	// Create mirror texture and an FBO used to copy mirror texture to back buffer
	ovrResult result = ovr_CreateMirrorTextureGL(session, &desc, &mirrorTexture);
	if (!OVR_SUCCESS(result))
	{
		ovrErrorInfo errorInfo;
		ovr_GetLastErrorInfo(&errorInfo);
		T_WARNING("%s", errorInfo.ErrorString);
		T_WARNING("Failed to create mirror texture.");
		return;
	}

	// Configure the mirror read buffer
	ovr_GetMirrorTextureBufferGL(session, mirrorTexture, &texId);
	m_context->functions()->glGenFramebuffers(1, &mirrorFBO);
	m_context->functions()->glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
	m_context->functions()->glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId, 0);
	m_context->functions()->glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
	m_context->functions()->glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

	// FloorLevel will give tracking poses where the floor height is 0
	ovr_SetTrackingOriginType(session, ovrTrackingOrigin_FloorLevel);

	// Initialize VR structures, filling out description.
	eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
	eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);
	hmdToEyeViewOffset[0] = eyeRenderDesc[0].HmdToEyeOffset;
	hmdToEyeViewOffset[1] = eyeRenderDesc[1].HmdToEyeOffset;


	// Initialize our single full screen Fov layer.
	layer.Header.Type = ovrLayerType_EyeFov;
	layer.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;
	layer.ColorTexture[0] = renderTexture->TextureChain;
	layer.ColorTexture[1] = renderTexture->TextureChain;
	layer.Fov[0] = eyeRenderDesc[0].Fov;
	layer.Fov[1] = eyeRenderDesc[1].Fov;
	layer.Viewport[0] = Recti(0, 0, bufferSize.w / 2, bufferSize.h);
	layer.Viewport[1] = Recti(bufferSize.w / 2, 0, bufferSize.w / 2, bufferSize.h);
	// ld.RenderPose and ld.SensorSampleTime are updated later per frame.
#else
	renderTexture = new TextureBuffer(session, true, true, bufferSize, NULL, m_context->functions());
	depthBuffer = new DepthBuffer(renderTexture->GetSize(), 0);
	if (!renderTexture->TextureSet)
	{
		T_WARNING("SwapTexture creation failed.");
		return;
	}


	ovrSizei windowSize = hmdDesc.Resolution;
	// Create mirror texture and an FBO used to copy mirror texture to back buffer
	ovrResult result = ovr_CreateMirrorTextureGL(session, GL_SRGB8_ALPHA8, windowSize.w, windowSize.h, reinterpret_cast<ovrTexture**>(&mirrorTexture));
	if (!OVR_SUCCESS(result))
	{
		T_WARNING("Failed to create mirror texture.");
		return;
	}

	// Configure the mirror read buffer
	m_context->functions()->glGenFramebuffers(1, &mirrorFBO);
	m_context->functions()->glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
	m_context->functions()->glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mirrorTexture->OGL.TexId, 0);
	m_context->functions()->glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
	m_context->functions()->glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

	// Initialize VR structures, filling out description.
	eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
	eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);
	hmdToEyeViewOffset[0] = eyeRenderDesc[0].HmdToEyeViewOffset;
	hmdToEyeViewOffset[1] = eyeRenderDesc[1].HmdToEyeViewOffset;


	// Initialize our single full screen Fov layer.
	layer.Header.Type = ovrLayerType_EyeFov;
	layer.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;
	layer.ColorTexture[0] = renderTexture->TextureSet;
	layer.ColorTexture[1] = renderTexture->TextureSet;
	layer.Fov[0] = eyeRenderDesc[0].Fov;
	layer.Fov[1] = eyeRenderDesc[1].Fov;
	layer.Viewport[0] = Recti(0, 0, bufferSize.w / 2, bufferSize.h);
	layer.Viewport[1] = Recti(bufferSize.w / 2, 0, bufferSize.w / 2, bufferSize.h);
	// ld.RenderPose and ld.SensorSampleTime are updated later per frame.
#endif

	//Configure mirror fbo for qml
	m_context->functions()->glGenTextures(1, &mirrorTextureForQML);
	m_context->functions()->glBindTexture(GL_TEXTURE_2D, mirrorTextureForQML);
	m_context->functions()->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_context->functions()->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_context->functions()->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_context->functions()->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_context->functions()->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, renderTexture->texSize.w, renderTexture->texSize.h, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);

	// load textures and create framebuffers
	m_context->functions()->glGenFramebuffers(1, &mirrorFBOForQML);
	m_context->functions()->glBindFramebuffer(GL_FRAMEBUFFER, mirrorFBOForQML);
	m_context->functions()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mirrorTextureForQML, 0);
	m_context->functions()->glBindFramebuffer(GL_FRAMEBUFFER, 0);


	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	GPUProgram::ADD_SHADER_FROM_CODE(m_program, "vert", "Oculus");
	GPUProgram::ADD_SHADER_FROM_CODE(m_program, "frag", "Oculus");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/Oculus.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/Oculus.frag");
#endif
	m_program->link();
	m_vertexAttr = m_program->attributeLocation("vertex");
	m_texCoordAttr = m_program->attributeLocation("texCoord");
	m_matrixUnif = m_program->uniformLocation("matrix");
	m_isRiftUnif = m_program->uniformLocation("bForRift");

	m_program->bind();
	m_context->functions()->glUniform1i(m_program->uniformLocation("texture"), 0);
	m_program->release();

#if DEBUG_OCULUS_VIEWER
	m_gridProgram = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	GPUProgram::ADD_SHADER_FROM_CODE(m_gridProgram, "vert", "OculusGrid");
	GPUProgram::ADD_SHADER_FROM_CODE(m_gridProgram, "frag", "OculusGrid");
#endif
	m_gridProgram->link();
	m_gridVertexAttr = m_gridProgram->attributeLocation("vertex");
	m_gridMatrixUnif = m_gridProgram->uniformLocation("matrix");
#endif

	m_context->doneCurrent();

	// Create a unit sphere
	vertices = new GLfloat[VERT_SEGS*HORZ_SEGS * 6 * 3];
	texCoords = new GLfloat[VERT_SEGS*HORZ_SEGS * 6 * 2];
	texCoordsRight = new GLfloat[VERT_SEGS*HORZ_SEGS * 6 * 2];
	for (int isS = 0; isS < VERT_SEGS; isS++)
	{
		float y1 = 1.0f * isS / VERT_SEGS;
		float y2 = 1.0f * (isS + 1) / VERT_SEGS;
		float phi1 = (1.0f - 2.0f * y1) * PO_PI / 2.0f;
		float phi2 = (1.0f - 2.0f * y2) * PO_PI / 2.0f;
		for (int j = 0; j < HORZ_SEGS; j++)
		{
			float x1 = 1.0f * j / HORZ_SEGS;
			float x2 = 1.0f * (j + 1) / HORZ_SEGS;

			float theta1 = (2.0f * x1 - 1.0f) * PO_PI;
			float theta2 = (2.0f * x2 - 1.0f) * PO_PI;

			float x11 = cosf(phi1) * sinf(theta1);
			float y11 = sinf(phi1);
			float z11 = -cosf(phi1) * cosf(theta1);

			float x12 = cosf(phi1) * sinf(theta2);
			float y12 = sinf(phi1);
			float z12 = -cosf(phi1) * cosf(theta2);

			float x21 = cosf(phi2) * sinf(theta1);
			float y21 = sinf(phi2);
			float z21 = -cosf(phi2) * cosf(theta1);

			float x22 = cosf(phi2) * sinf(theta2);
			float y22 = sinf(phi2);
			float z22 = -cosf(phi2) * cosf(theta2);

			// 11 - 12
			//    /
			// 21
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 0 * 3 + 0] = x11;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 0 * 3 + 1] = y11;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 0 * 3 + 2] = z11;

			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 1 * 3 + 0] = x12;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 1 * 3 + 1] = y12;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 1 * 3 + 2] = z12;

			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 2 * 3 + 0] = x21;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 2 * 3 + 1] = y21;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 2 * 3 + 2] = z21;

			if (!isStereo)
			{
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 0 * 2 + 0] = x1;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 0 * 2 + 1] = y1;

				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 1 * 2 + 0] = x2;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 1 * 2 + 1] = y1;

				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 2 * 2 + 0] = x1;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 2 * 2 + 1] = y2;
			}
			else
			{
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 0 * 2 + 0] = x1;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 0 * 2 + 1] = y1 / 2;

				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 1 * 2 + 0] = x2;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 1 * 2 + 1] = y1 / 2;

				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 2 * 2 + 0] = x1;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 2 * 2 + 1] = y2 / 2;

				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 0 * 2 + 0] = x1;
				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 0 * 2 + 1] = 0.5f + y1 / 2;

				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 1 * 2 + 0] = x2;
				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 1 * 2 + 1] = 0.5f + y1 / 2;

				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 2 * 2 + 0] = x1;
				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 2 * 2 + 1] = 0.5f + y2 / 2;
			}

			// 21 - 12
			//    /
			// 22
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 3 * 3 + 0] = x21;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 3 * 3 + 1] = y21;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 3 * 3 + 2] = z21;

			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 4 * 3 + 0] = x12;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 4 * 3 + 1] = y12;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 4 * 3 + 2] = z12;

			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 5 * 3 + 0] = x22;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 5 * 3 + 1] = y22;
			vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 5 * 3 + 2] = z22;

			if (!isStereo)
			{
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 3 * 2 + 0] = x1;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 3 * 2 + 1] = y2;

				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 4 * 2 + 0] = x2;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 4 * 2 + 1] = y1;

				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 5 * 2 + 0] = x2;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 5 * 2 + 1] = y2;
			}
			else
			{
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 3 * 2 + 0] = x1;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 3 * 2 + 1] = y2 / 2;

				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 4 * 2 + 0] = x2;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 4 * 2 + 1] = y1 / 2;

				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 5 * 2 + 0] = x2;
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 5 * 2 + 1] = y2 / 2;

				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 3 * 2 + 0] = x1;
				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 3 * 2 + 1] = 0.5f + y2 / 2;

				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 4 * 2 + 0] = x2;
				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 4 * 2 + 1] = 0.5f + y1 / 2;

				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 5 * 2 + 0] = x2;
				texCoordsRight[(isS * HORZ_SEGS + j) * 6 * 2 + 5 * 2 + 1] = 0.5f + y2 / 2;
			}
		}
	}

	verticesForQML = new GLfloat[6 * 3];
	texCoordsForQML = new GLfloat[6 * 2];

	verticesForQML[0] = -1; verticesForQML[1] = -1;
	verticesForQML[3] = -1; verticesForQML[4] = 1;
	verticesForQML[6] = 1; verticesForQML[7] = 1;

	verticesForQML[9] = -1; verticesForQML[10] = -1;
	verticesForQML[12] = 1; verticesForQML[13] = 1;
	verticesForQML[15] = 1; verticesForQML[16] = -1;
	verticesForQML[2] = verticesForQML[5] = verticesForQML[8] = verticesForQML[11] = verticesForQML[14] = verticesForQML[17] = 0;

	texCoordsForQML[0] = 0; texCoordsForQML[1] = 0;
	texCoordsForQML[2] = 0; texCoordsForQML[3] = 1;
	texCoordsForQML[4] = 1; texCoordsForQML[5] = 1;

	texCoordsForQML[6] = 0; texCoordsForQML[7] = 0;
	texCoordsForQML[8] = 1; texCoordsForQML[9] = 1;
	texCoordsForQML[10] = 1; texCoordsForQML[11] = 0;

#if DEBUG_OCULUS_VIEWER
	// 2 * corners for one direction
	// * 2 for rectangle
	// * 4 for 4 face
	m_gridPointCount = 2 * OCULUS_VIEWER_GRID_CORNERS * 2;
	verticesGrid = new GLfloat[m_gridPointCount * 3];
	float grid_size = 1;
	float step_size = 1.0f / (OCULUS_VIEWER_GRID_CORNERS - 1);
	int nOffset = 0;
	for (int i = 0; i < OCULUS_VIEWER_GRID_CORNERS; i++)
	{
		float z = -1;
		float y1 = -1;
		float y2 = 1;
		float x = -1 + i * step_size * 2.0f;
		// z = +-1
		verticesGrid[nOffset + i * 3 * 2 + 0 * 3 + 0] = x;
		verticesGrid[nOffset + i * 3 * 2 + 0 * 3 + 1] = y1;
		verticesGrid[nOffset + i * 3 * 2 + 0 * 3 + 2] = z;

		verticesGrid[nOffset + i * 3 * 2 + 1 * 3 + 0] = x;
		verticesGrid[nOffset + i * 3 * 2 + 1 * 3 + 1] = y2;
		verticesGrid[nOffset + i * 3 * 2 + 1 * 3 + 2] = z;
	}
	nOffset = nOffset + OCULUS_VIEWER_GRID_CORNERS * 3 * 2;

	for (int i = 0; i < OCULUS_VIEWER_GRID_CORNERS; i++)
	{
		float z = -1;
		float x1 = -1;
		float x2 = 1;
		float y = -1 + i * step_size * 2.0f;
		// z = +-1
		verticesGrid[nOffset + i * 3 * 2 + 0 * 3 + 0] = x1;
		verticesGrid[nOffset + i * 3 * 2 + 0 * 3 + 1] = y;
		verticesGrid[nOffset + i * 3 * 2 + 0 * 3 + 2] = z;

		verticesGrid[nOffset + i * 3 * 2 + 1 * 3 + 0] = x2;
		verticesGrid[nOffset + i * 3 * 2 + 1 * 3 + 1] = y;
		verticesGrid[nOffset + i * 3 * 2 + 1 * 3 + 2] = z;
	}
	
	for (int i = 0; i < m_gridPointCount * 3; i++)
	{
		verticesGrid[i] = verticesGrid[i] * grid_size;
	}
#endif
	
	//QTimer *timer = new QTimer(this);
	//connect(timer, SIGNAL(timeout()), this, SLOT(renderFrame()));
	//timer->start(1000 / 75.0f);
	//timer->start(1);
}


void
OculusRender::resizeGL(int width, int height)
{
}

int OculusRender::getTextureID()
{
#ifndef OCULUS_1_8_0
	int texId = (int)mirrorTexture->OGL.TexId;
#endif
	return texId;
}

int OculusRender::getTextureIDForQML()
{
	return mirrorTextureForQML;
}


void
OculusRender::paintGL()
{
	//renderFrame();
}

void OculusRender::renderFrame()
{
	if (!m_isCreated) return;

	m_context->makeCurrent(this);

#ifdef OCULUS_1_8_0
	ovrSessionStatus sessionStatus;
	ovr_GetSessionStatus(session, &sessionStatus);
	if (sessionStatus.ShouldQuit)
	{
		// Because the application is requested to quit, should not request retry
		return;
	}
	if (sessionStatus.ShouldRecenter)
		ovr_RecenterTrackingOrigin(session);

	if (sessionStatus.IsVisible)
	{

	}
#endif

	// 3. Integrate head-tracking into application's view and movement code.
	// a. Obtaining predicted headset orientation for the frame through a combination of the
	// GetPredictedDisplayTime and ovr_GetTrackingState calls.

	double ftiming = ovr_GetPredictedDisplayTime(session, 0);
#ifdef OCULUS_1_8_0
	double sensorSampleTime; // sensorSampleTime is fed into the layer later
	ovr_GetEyePoses(session, frameIndex, ovrTrue, hmdToEyeViewOffset, layer.RenderPose, &sensorSampleTime);
#else
	double sensorSampleTime = ovr_GetTimeInSeconds();
	ovrTrackingState hmdState = ovr_GetTrackingState(session, ftiming, ovrTrue);
	ovr_CalcEyePoses(hmdState.HeadPose.ThePose, hmdToEyeViewOffset, layer.RenderPose);
#endif

	layer.SensorSampleTime = sensorSampleTime;

	Vector3f originPos;
	Matrix4f originRot;

	//if (isVisible)
	{
		// Increment to use next texture, just before writing
#ifndef OCULUS_1_8_0
		renderTexture->TextureSet->CurrentIndex = (renderTexture->TextureSet->CurrentIndex + 1) % renderTexture->TextureSet->TextureCount;
#endif
		// Switch to eye render target
		renderTexture->SetAndClearRenderSurface(depthBuffer);

		glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// b. Applying Rift orientation and position to the camera view, while combining it with other application
		// controls.
		// c.Modifying movement and game play to consider head orientation.
		for (int eye = 0; eye < 2; ++eye)
		{
			// Get view and projection matrices
			Vector3f pos;
			/*if (isStereo)
				pos = originPos;
			else*/
				pos = originPos + originRot.Transform(layer.RenderPose[eye].Position);
			Matrix4f rot = originRot;
			if (((Quat<float>)layer.RenderPose[eye].Orientation).IsNormalized())
				rot = rot * Matrix4f(layer.RenderPose[eye].Orientation);
			Vector3f finalUp = rot.Transform(Vector3f(0, 1, 0));
			Vector3f finalForward = rot.Transform(Vector3f(0, 0, -1));
			Matrix4f view = Matrix4f::LookAtRH(pos, pos + finalForward, finalUp);
			Matrix4f proj = ovrMatrix4f_Projection(layer.Fov[eye], 0.2f, 1000.0f,
#ifndef OCULUS_1_8_0
				ovrProjection_RightHanded);
#else
				ovrProjection_None);
#endif

			// Render world
			Recti viewport = layer.Viewport[eye];
			glViewport(viewport.x, viewport.y, viewport.w, viewport.h);
			Render(view, proj, eye);
		}

		// Avoids an error when calling SetAndClearRenderSurface during next iteration.
		// Without this, during the next while loop iteration SetAndClearRenderSurface
		// would bind a framebuffer with an invalid COLOR_ATTACHMENT0 because the texture ID
		// associated with COLOR_ATTACHMENT0 had been unlocked by calling wglDXUnlockObjectsNV.
		renderTexture->UnsetRenderSurface();
#ifdef OCULUS_1_8_0
		// Commit changes to the textures so they get picked up frame
		renderTexture->Commit();
#endif
	}

	//For QML

	m_context->functions()->glBindFramebuffer(GL_FRAMEBUFFER, mirrorFBOForQML);
	m_context->functions()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mirrorTextureForQML, 0);

	m_context->functions()->glViewport(0, 0, renderTexture->texSize.w, renderTexture->texSize.h);
	m_context->functions()->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	Matrix4f matId = Matrix4f::Identity();
	RenderForQMLTexture(matId);
	m_context->functions()->glBindFramebuffer(GL_FRAMEBUFFER, mirrorFBOForQML);
	m_context->functions()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	

#ifdef OCULUS_1_8_0
	layer.SensorSampleTime = sensorSampleTime;
#else
	// Set up positional data.
	ovrViewScaleDesc viewScaleDesc;
	viewScaleDesc.HmdSpaceToWorldScaleInMeters = 1.0f;
	viewScaleDesc.HmdToEyeViewOffset[0] = hmdToEyeViewOffset[0];
	viewScaleDesc.HmdToEyeViewOffset[1] = hmdToEyeViewOffset[1];
#endif

	// Submit frame with one layer we have.
	ovrLayerHeader* layers = &layer.Header;
#ifdef OCULUS_1_8_0
	ovrResult result = ovr_SubmitFrame(session, frameIndex, nullptr, &layers, 1);
#else
	//ovrResult result = ovr_SubmitFrame(HMD, 0, &viewScaleDesc, &layers, 1);
	ovrResult result = ovr_SubmitFrame(session, 0, nullptr, &layers, 1);
#endif

	if (!OVR_SUCCESS(result))
		;// goto Done;

#ifdef OCULUS_1_8_0
	frameIndex++;
#endif

	// exit the rendering loop if submit returns an error, will retry on ovrError_DisplayLost

	// Blit mirror texture to back buffer
	m_context->functions()->glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
	m_context->functions()->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
#ifdef OCULUS_1_8_0
	GLint w = renderTexture->texSize.w;
	GLint h = renderTexture->texSize.h;
#else
	GLint w = mirrorTexture->OGL.Header.TextureSize.w;
	GLint h = mirrorTexture->OGL.Header.TextureSize.h;
#endif
	((QOpenGLFunctions_4_3_Core*)m_context->versionFunctions())->glBlitFramebuffer(0, h, w, 0,
		0, 0, w, h,
		GL_COLOR_BUFFER_BIT, GL_LINEAR);
	m_context->functions()->glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

// 	m_context->swapBuffers(this);
	m_context->doneCurrent();
	//glEnable(GL_FRAMEBUFFER_SRGB);
}

void OculusRender::RenderForQMLTexture(Matrix4f viewproj)
{
	m_program->bind();

	Matrix4f matrix = viewproj;
	float matrixValue[16];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++)
			matrixValue[i * 4 + j] = matrix.M[i][j];
	}
	m_context->functions()->glUniformMatrix4fv(m_matrixUnif, 1, true, matrixValue);
	m_context->functions()->glActiveTexture(GL_TEXTURE0);
#ifdef OCULUS_1_8_0
	m_context->functions()->glBindTexture(GL_TEXTURE_2D, texId);
#else
	m_context->functions()->glBindTexture(GL_TEXTURE_2D, mirrorTexture->OGL.TexId);
#endif

	// this is for QMLInteractiveView in Oculus.frag
	m_context->functions()->glUniform1i(m_isRiftUnif, 0);

	m_context->functions()->glVertexAttribPointer(m_vertexAttr, 3, GL_FLOAT, GL_FALSE, 0, verticesForQML);
	
	m_context->functions()->glVertexAttribPointer(m_texCoordAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoordsForQML);

	m_context->functions()->glEnableVertexAttribArray(m_vertexAttr);
	m_context->functions()->glEnableVertexAttribArray(m_texCoordAttr);

	m_context->functions()->glDrawArrays(GL_TRIANGLES, 0, 6);

	m_context->functions()->glDisableVertexAttribArray(m_texCoordAttr);
	m_context->functions()->glDisableVertexAttribArray(m_vertexAttr);


	m_program->release();
}

void OculusRender::Render(Matrix4f view, Matrix4f proj, int isRight)
{
	m_program->bind();

	Matrix4f matrix = proj*view*Matrix4f::Scaling(10.0f);
	float matrixValue[16];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++)
			matrixValue[i * 4 + j] = matrix.M[i][j];
	}
	m_context->functions()->glUniformMatrix4fv(m_matrixUnif, 1, true, matrixValue);

	m_context->functions()->glActiveTexture(GL_TEXTURE0);
	m_context->functions()->glBindTexture(GL_TEXTURE_2D, m_panoramaTexture);
	

	// this is for oculus rift in Oculus.frag
	m_context->functions()->glUniform1i(m_isRiftUnif, 1);
	


	m_context->functions()->glVertexAttribPointer(m_vertexAttr, 3, GL_FLOAT, GL_FALSE, 0, vertices);
	if (isStereo)
	{
		if (isRight)
		{
			m_context->functions()->glVertexAttribPointer(m_texCoordAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoordsRight);
		}
		else
		{
			m_context->functions()->glVertexAttribPointer(m_texCoordAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords);
		}
	}
	else
		m_context->functions()->glVertexAttribPointer(m_texCoordAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords);

	m_context->functions()->glEnableVertexAttribArray(m_vertexAttr);
	m_context->functions()->glEnableVertexAttribArray(m_texCoordAttr);

	m_context->functions()->glDrawArrays(GL_TRIANGLES, 0, HORZ_SEGS * VERT_SEGS * 6);

	m_context->functions()->glDisableVertexAttribArray(m_texCoordAttr);
	m_context->functions()->glDisableVertexAttribArray(m_vertexAttr);

	m_program->release();

	
#if DEBUG_OCULUS_VIEWER
	if (isChessboard)
	{
		m_gridProgram->bind();

		m_context->functions()->glUniformMatrix4fv(m_gridMatrixUnif, 1, true, matrixValue);
		// Draw Grid
		((QOpenGLFunctions_4_3_Core*)m_context->versionFunctions())->glLineWidth(3);
		((QOpenGLFunctions_4_3_Core*)m_context->versionFunctions())->glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		m_context->functions()->glVertexAttribPointer(m_gridVertexAttr, 3, GL_FLOAT, GL_FALSE, 0, verticesGrid);
		m_context->functions()->glEnableVertexAttribArray(m_gridVertexAttr);

		m_context->functions()->glDrawArrays(GL_LINES, 0, m_gridPointCount);

		m_context->functions()->glDisableVertexAttribArray(m_gridVertexAttr);

		((QOpenGLFunctions_4_3_Core*)m_context->versionFunctions())->glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		m_gridProgram->release();
	}
#endif
}

void OculusRender::run()
{
	while (1)
	{
		if (QCoreApplication::hasPendingEvents())
			QCoreApplication::processEvents();

		if (QThread::currentThread()->isInterruptionRequested())
		{
			PANO_LOG("Got signal to terminate");
			doStop = true;
		}
		//
		// Stop thread if doStop = TRUE 
		//
		doStopMutex.lock();
		if (doStop)
		{
			PANO_LOG("Stop");
			doStop = false;
			doStopMutex.unlock();
			break;
		}
		doStopMutex.unlock();

		// Capture frame (if available)
		renderFrame();
	}

	PANO_LOG("Stopping oculus thread...");
}

void OculusRender::process()
{
	m_finished = false;
	run();
	isDisconnecting = true;
	PANO_LOG("Oculus Thread - Emit finished signal");
	finishWC.wakeAll();
	if (oculusThreadInstance)
	{
		delete oculusThreadInstance;	// This will be deleted automatically by deleteLater()
		oculusThreadInstance = NULL;
	}
	if (stopReason == SWITCH_OFF)
		emit finished(THREAD_TYPE_OCULUS, MSG_OCULUS_SWITCHOFF, -1);
	else
		emit finished(THREAD_TYPE_OCULUS, "", -1);
}

bool OculusRender::isFinished()
{
	return m_finished;
}

bool OculusRender::disconnect()
{
	//
	// Camera is connected
	//
	if (m_isCreated)
	{
		PANO_LOG("Disconnecting Oculus Rift");
		if (m_finished == false) {
			this->thread()->exit();
			if (!isDisconnecting)
			{
				QMutex finishMutex;
				finishMutex.lock();
				finishWC.wait(&finishMutex, 1000);
				finishMutex.unlock();
			}
		}
		//
		// Disconnect camera
		//
		m_context->doneCurrent();
		m_context->moveToThread(QApplication::instance()->thread());
		m_context->makeCurrent(this);
		if (vertices)
		{
			delete[] vertices;
			vertices = 0;
		}

		if (texCoords)
		{
			delete[] texCoords;
			texCoords = 0;
		}

		if (texCoordsRight)
		{
			delete[] texCoordsRight;
			texCoordsRight = 0;
		}

		if (verticesForQML)
		{
			delete[] verticesForQML;
			verticesForQML = 0;
		}

		if (texCoordsForQML)
		{
			delete[] texCoordsForQML;
			texCoordsForQML = 0;
		}

#if DEBUG_OCULUS_VIEWER
		if (verticesGrid)
		{
			delete[] verticesGrid;
			verticesGrid = 0;
		}
#endif

		if (renderTexture)
		{
			delete renderTexture;
			renderTexture = nullptr;
		}
		if (depthBuffer)
		{
			delete depthBuffer;
			depthBuffer = nullptr;
		}

		PANO_LOG("Oculus SDK shut down");
		return true;
	}
	else
	{
		//
		// Camera is NOT connected
		//
		return false;
	}	
}

void OculusRender::startThread()
{
	oculusThreadInstance = new QThread();
	//this->moveToThread(oculusThreadInstance);
	QObject::connect(oculusThreadInstance, SIGNAL(started()), this, SLOT(process()));
	QObject::connect(oculusThreadInstance, SIGNAL(finished()), this, SLOT(deleteLater()));
	QObject::connect(this, SIGNAL(finished(int, QString, int)), oculusThreadInstance, SLOT(deleteLater()));

	m_context->doneCurrent();
	//m_context->moveToThread(oculusThreadInstance);
	oculusThreadInstance->start();
	oculusThreadInstance->setPriority(QThread::HighestPriority);
}

void OculusRender::stop(OculusRender::StopReason reason)
{
	QMutexLocker locker(&doStopMutex);
	stopReason = reason;
	doStop = true;
}

bool OculusRender::isConnected()
{
	return m_isCreated;
}