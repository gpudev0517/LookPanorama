#include "GLSLFinalPanorama.h"
#include "define.h"
#include "common.h"

GLSLPanorama::GLSLPanorama(QObject *parent) : GPUPanorama(parent)
{
}

GLSLPanorama::~GLSLPanorama()
{
}

void GLSLPanorama::initialize(int panoWidth, int panoHeight, bool isStereo)
{
	this->m_stereo = isStereo;
	this->panoramaViewWidth = panoWidth;
	if (m_stereo)
	{
		this->panoramaViewHeight = panoHeight * 2;
	}
	else
	{
		this->panoramaViewHeight = panoHeight;
	}

	// frame buffer
	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, panoramaViewWidth, panoramaViewHeight, 0, GL_ALPHA, GL_UNSIGNED_BYTE, NULL);

	// load textures and create framebuffers
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "panoView");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/panoView.frag");
#endif
	m_program->link();
	m_program->bind();

	int textureIds[2] = { 0, 1 };
	m_gl->glUniform1iv(m_program->uniformLocation(QString("textures")), 2, textureIds);
	m_gl->glUniform1i(m_program->uniformLocation(QString("isStereo")), isStereo);
	m_program->release();

	m_initialized = true;
}


void GLSLPanorama::render(GPUResourceHandle fbos[], bool isOutput)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = panoramaViewWidth;
	float height = panoramaViewHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	int panoCount = m_stereo ? 2 : 1;
	for (int i = 0; i < panoCount; i++)
	{
		m_gl->glActiveTexture(GL_TEXTURE0 + i);
		m_gl->glBindTexture(GL_TEXTURE_2D, fbos[i]);
	}

	//if (isOutput)
	//	refineTexCoordsForOutput(texCoords, 4);
	m_program->setUniformValue("isOutput", isOutput);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
	m_gl->glFinish();

	m_program->release();

#if DEBUG_SHADER
	QString strSaveName = "finalpanorama.png";
	saveTexture(m_fboId, getWidth(), getHeight(), strSaveName, m_gl, false);
#endif 
}

void GLSLPanorama::downloadTexture(unsigned char* alphaBuffer)
{
	if (m_initialized) {
		m_program->bind();
		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
		m_gl->glFinish();

		m_functions_2_0->glReadPixels(0, 0, getWidth(), getHeight(), GL_ALPHA, GL_UNSIGNED_BYTE, alphaBuffer);
		
		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
		m_program->release();
	}
}

// Final Panorama
GLSLFinalPanorama::GLSLFinalPanorama(QObject *parent) : GPUFinalPanorama(parent)
{
}

GLSLFinalPanorama::~GLSLFinalPanorama()
{
	if (m_initialized)
	{
		m_gl->glDeleteTextures(2, m_fboTextureIds);
		m_gl->glDeleteFramebuffers(2, m_fboIds);
		for (int i = 0; i < 2; i++)
		{
			m_pbos[i]->destroy();
			delete m_pbos[i];
			m_pbos[i] = NULL;
		}
	}
}

void GLSLFinalPanorama::initialize(int panoWidth, int panoHeight, bool isStereo)
{
	// frame buffer
	m_gl->glGenTextures(2, m_fboTextureIds);
	m_gl->glGenFramebuffers(2, m_fboIds);
	for (int i = 0; i < 2; i++)
	{
		m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureIds[i]);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		// load textures and create framebuffers
		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[i]);
		m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureIds[i], 0);
	}
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "finalPanoView");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/finalPanoView.frag");
#endif
	m_program->link();
	m_program->bind();

	m_program->setUniformValue("texture", 0);

	for (int i = 0; i < 2; i++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelPackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::StreamRead);
		m_pbos[i] = pbo;
	}

	reconfig(FinalPanoramaConfig(panoWidth, isStereo?panoHeight*2:panoHeight, isStereo));
	m_initialized = true;
}

void GLSLFinalPanorama::reconfig(FinalPanoramaConfig config)
{
	int newPanoHeight = config.panoHeight;
	newPanoHeight = newPanoHeight * 3 / 2;

	if (config.panoWidth == panoramaViewWidth && newPanoHeight == panoramaViewHeight && config.isStereo == m_stereo)
		return;
	this->m_stereo = config.isStereo;
	this->panoramaViewWidth = config.panoWidth;
	this->panoramaViewHeight = newPanoHeight;

	for (int i = 0; i < 2; i++)
	{
		m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureIds[i]);
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, panoramaViewWidth, panoramaViewHeight, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
	}
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
	m_workingGPUResource = 0;

	for (int i = 0; i < 2; i++)
	{
		m_pbos[i]->bind();
		m_pbos[i]->allocate(panoramaViewWidth * panoramaViewHeight);
		m_pbos[i]->release();
	}
	m_targetArrayIndex = 0;
}

void GLSLFinalPanorama::render(GPUResourceHandle fbo)
{
	if (newConfig.size() != 0)
	{
		reconfig(newConfig[0]);
		newConfig.clear();
	}
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[m_workingGPUResource]);

	float width = panoramaViewWidth;
	float height = panoramaViewHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, fbo);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_workingGPUResource = 1 - m_workingGPUResource;

	m_program->release();

#if DEBUG_SHADER
	QString strSaveName = "finalpanorama.png";
	saveTexture(m_fboId, getWidth(), getHeight(), strSaveName, m_gl);
#endif 
}

void GLSLFinalPanorama::downloadTexture(unsigned char* rgbBuffer)
{
	if (m_initialized) {
		m_program->bind();
		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[1 - m_workingGPUResource]);

		m_targetArrayIndex = (m_targetArrayIndex + 1) % 2;
		QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
		QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % 2];

		pboCurrent->bind();
		m_functions_2_0->glReadPixels(0, 0, getWidth(), getHeight(), GL_RED, GL_UNSIGNED_BYTE, 0);
		pboCurrent->release();

		pboNext->bind();
		GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::ReadOnly);
		if (ptr)
		{
			memcpy(rgbBuffer, ptr, getWidth() * getHeight());
			pboNext->unmap();
		}
		pboNext->release();

		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
		m_program->release();
	}
}
