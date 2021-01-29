#include "GLSLUnwarp.h"
//#include "QmlMainWindow.h"

//extern QmlMainWindow* g_mainWindow;

GLSLUnwarp::GLSLUnwarp(QObject *parent, bool isYUV) : GPUUnwarp(parent)
{
}

GLSLUnwarp::~GLSLUnwarp()
{
	if (m_initialized)
	{
		m_functions_4_3->glDeleteBuffers(1, &ubo);
	}
	
}

void GLSLUnwarp::initialize(int id,  int xres, int yres, int panoWidth, int panoHeight)
{
	camID = id;
	panoramaWidth = panoWidth;
	panoramaHeight = panoHeight;

	inputWidth = xres;
	inputHeight = yres;

	// frame buffer
	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, panoramaWidth, panoramaHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "stitcher");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/stitcher.frag");
#endif
	m_program->link();

	m_program->bind();

	m_functions_4_3->glGenBuffers(1, &ubo);

	m_program->setUniformValue("texture", 0);
	m_functions_4_3->glGenBuffers(1, &ubo);

	m_program->release();

	m_initialized = true;
}

void GLSLUnwarp::updateCameraParams()
{
	// camera parameters

	float nwidth = camInput.xres;
	float nheight = camInput.yres;

	cam.lens = camInput.m_cameraParams.m_lensType;
	cam.imageWidth = nwidth;
	cam.imageHeight = nheight;
	cam.cx = cam.imageWidth / 2;
	cam.cy = cam.imageHeight / 2;
	cam.FoV = camInput.m_cameraParams.m_fov;
	cam.FoVY = camInput.m_cameraParams.m_fovy;
	cam.offset_x = camInput.m_cameraParams.m_offset_x;
	cam.offset_y = camInput.m_cameraParams.m_offset_y;

	cam.k1 = camInput.m_cameraParams.m_k1;
	cam.k2 = camInput.m_cameraParams.m_k2;
	cam.k3 = camInput.m_cameraParams.m_k3;
}

void GLSLUnwarp::render(GPUResourceHandle rgbTextureId, RenderMode renderMode)
{
#if 0
	QMatrix3x3 mYaw = getViewMatrix(setting.m_fYaw, 0, 0);
	QMatrix3x3 mPitch = getViewMatrix(0, setting.m_fPitch, 0);
	QMatrix3x3 mRoll = getViewMatrix(0, 0, setting.m_fRoll);
	QMatrix3x3 m2 = getViewMatrix(cam.m_yaw, cam.m_pitch, cam.m_roll);
	QMatrix3x3 m = m2 * mYaw * mPitch * mRoll;
	m_program->setUniformValue(cpUnif, m);
#else
	mat3 m = getCameraViewMatrix(camInput.m_cameraParams.m_yaw, camInput.m_cameraParams.m_pitch, camInput.m_cameraParams.m_roll);
	cam.cP.set_mat3(m);
#endif
	updateCameraParams();

	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, panoramaWidth, panoramaHeight);
	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, rgbTextureId);
	
	GLuint uboIndex = m_functions_4_3->glGetUniformBlockIndex(m_program->programId(), "cameraBuffer"); // get index of block
	m_functions_4_3->glBindBuffer(GL_UNIFORM_BUFFER, ubo);
	m_functions_4_3->glBufferData(GL_UNIFORM_BUFFER, sizeof(CameraData),
		&cam, GL_STATIC_DRAW);
	m_functions_4_3->glBindBufferBase(GL_UNIFORM_BUFFER, uboIndex, ubo);

	m_program->setUniformValue("renderMode", renderMode);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();

#if DEBUG_SHADER
	QString strSaveName = QString("unwarp_") + (renderMode == Color ? "color_" : "weightmap_") + QString::number(camID) + ".png";
	saveTexture(m_fboId, panoramaWidth, panoramaHeight, strSaveName, m_gl);
#endif 
}