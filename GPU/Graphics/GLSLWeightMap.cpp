#include "GLSLWeightMap.h"
#include "define.h"
#include "common.h"
#include "define.h"

#define DELTA_SCALE 3

//Camera Weight Map
GLSLCameraWeightMap::GLSLCameraWeightMap(QObject *parent) : GPUCameraWeightMap(parent)
{
}

GLSLCameraWeightMap::~GLSLCameraWeightMap()
{
}

void GLSLCameraWeightMap::initialize(int imageWidth, int imageHeight)
{
	this->imageWidth = imageWidth;
	this->imageHeight = imageHeight;

	// frame buffer
	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, imageWidth, imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "cameraWeight");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/cameraWeight.frag");
#endif
	m_program->link();

	m_program->bind();
	//m_program->setUniformValue("texture", 0);

	m_program->setUniformValue("lens", camInput.m_cameraParams.m_lensType);
	m_program->setUniformValue("editedWeightTexture", 0);

	if (camInput.m_cameraParams.isFisheye())
		m_program->setUniformValue("blendingFalloff", 1.0f);
	else
		m_program->setUniformValue("blendingFalloff", 0.5f);

	// camera parameters

	float nWidth = imageWidth;
	float nHeight = imageHeight;
	m_program->setUniformValue("imageWidth", nWidth);
	m_program->setUniformValue("imageHeight", nHeight);

	xrad1Unif = m_program->uniformLocation("xrad1");
	xrad2Unif = m_program->uniformLocation("xrad2");
	yrad1Unif = m_program->uniformLocation("yrad1");
	yrad2Unif = m_program->uniformLocation("yrad2");


	m_program->release();

	m_initialized = true;
}

void GLSLCameraWeightMap::updateCameraParams()
{
	// camera parameters
	m_program->bind();

	m_program->setUniformValue("lens", camInput.m_cameraParams.m_lensType);

	float nwidth = camInput.xres;
	float nheight = camInput.yres;
	m_program->setUniformValue("imageWidth", nwidth);
	m_program->setUniformValue("imageHeight", nheight);

	m_program->release();
}

void GLSLCameraWeightMap::render(int camID)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = imageWidth;
	float height = imageHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

// 	m_gl->glActiveTexture(GL_TEXTURE0);
// 	m_gl->glBindTexture(GL_TEXTURE_2D, editedWeightTexId);
// 	

	// blending parameters
	// 8 camera set
	if (camInput.m_cameraParams.isFisheye())
	{
		m_program->setUniformValue(xrad1Unif, camInput.m_cameraParams.m_xrad1);
		m_program->setUniformValue(xrad2Unif, camInput.m_cameraParams.m_xrad2);
		m_program->setUniformValue(yrad1Unif, camInput.m_cameraParams.m_yrad1);
		m_program->setUniformValue(yrad2Unif, camInput.m_cameraParams.m_yrad2);
		m_program->setUniformValue("blendCurveStart", 0.4f);
	}
	else
	{
		// 6 camera set
		m_program->setUniformValue("fisheyeLensRadiusRatio1", camInput.m_cameraParams.m_xrad1);
		m_program->setUniformValue("fisheyeLensRadiusRatio2", camInput.m_cameraParams.m_yrad1);
		m_program->setUniformValue("blendCurveStart", 0.4f);
	}

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();

	GLenum error = glGetError(); 

#if DEBUG_SHADER
	QString strSaveName = QString("cameraweightmap_") + QString::number(camID) + ".png";
	saveTexture(m_fboId, width, height, strSaveName, m_gl);
#endif 
}


//Panorama Weight Map


GLSLPanoramaWeightMap::GLSLPanoramaWeightMap(QObject *parent, bool isYUV) : GPUPanoramaWeightMap(parent)
{
}

GLSLPanoramaWeightMap::~GLSLPanoramaWeightMap()
{
	if (m_initialized)
	{
		m_functions_4_3->glDeleteBuffers(1, &ubo);
	}
}

void GLSLPanoramaWeightMap::initialize(int xres, int yres, int panoWidth, int panoHeight)
{
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
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, panoramaWidth, panoramaHeight, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "panoramaWeight");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/panoramaWeight.frag");
#endif
	m_program->link();

	m_program->bind();

	m_functions_4_3->glGenBuffers(1, &ubo);
	m_program->setUniformValue("texture", 0);
	m_program->setUniformValue("editedWeightTexture", 1);

	m_program->release();

	m_initialized = true;
}

void GLSLPanoramaWeightMap::updateCameraParams()
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

void GLSLPanoramaWeightMap::render(unsigned int weightTextureId, unsigned int deltaWeightTexId, int camID)
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
	m_gl->glBindTexture(GL_TEXTURE_2D, weightTextureId);

	m_gl->glActiveTexture(GL_TEXTURE1);
	m_gl->glBindTexture(GL_TEXTURE_2D, deltaWeightTexId);
	
	GLuint uboIndex = m_functions_4_3->glGetUniformBlockIndex(m_program->programId(), "cameraBuffer"); // get index of block
	m_functions_4_3->glBindBuffer(GL_UNIFORM_BUFFER, ubo);
	m_functions_4_3->glBufferData(GL_UNIFORM_BUFFER, sizeof(CameraData),
		&cam, GL_STATIC_DRAW);
	m_functions_4_3->glBindBufferBase(GL_UNIFORM_BUFFER, uboIndex, ubo);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();

#if DEBUG_SHADER
	QString strSaveName = QString("panoramaweight_") + QString::number(camID) + ".png";
	saveTexture(m_fboId, panoramaWidth, panoramaHeight, strSaveName, m_gl);
#endif 
}
