#include "GLSLPyramidSum.h"
#include "define.h"

GLSLMergeTexture::GLSLMergeTexture(QObject *parent) : GPUProgram(parent)
{
}

GLSLMergeTexture::~GLSLMergeTexture()
{
}

void GLSLMergeTexture::initialize(int panoWidth, int panoHeight)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;

	// frame buffer
	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, panoramaWidth, panoramaHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "merge");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/merge.frag");
#endif
	m_program->link();
	m_program->bind();

	m_program->setUniformValue("texture1", 0);
	m_program->setUniformValue("texture2", 1);

	m_initialized = true;
}


void GLSLMergeTexture::render(GPUResourceHandle texture1, GPUResourceHandle texture2)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = panoramaWidth;
	float height = panoramaHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, texture1);
	m_gl->glActiveTexture(GL_TEXTURE1);
	m_gl->glBindTexture(GL_TEXTURE_2D, texture2);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();
}

void GLSLMergeTexture::clear()
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = panoramaWidth;
	float height = panoramaHeight;

	//m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClearColor(0.5f, 0.5f, 0.5f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();
}

const int GLSLMergeTexture::getWidth()
{
	return panoramaWidth;
}

const int GLSLMergeTexture::getHeight()
{
	return panoramaHeight;
}
////////////////

GLSLPyramidSum::GLSLPyramidSum(QObject *parent) : GPUProgram(parent)
, m_merge1(NULL)
, m_merge2(NULL)
{
}

GLSLPyramidSum::~GLSLPyramidSum()
{
	if (m_merge1)
	{
		delete m_merge1;
		m_merge1 = NULL;
	}
	if (m_merge2)
	{
		delete m_merge2;
		m_merge2 = NULL;
	}
}

void GLSLPyramidSum::initialize(int panoWidth, int panoHeight)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;

	m_merge1 = new GLSLMergeTexture();
	m_merge1->setGL(m_gl, m_functions_2_0);
	m_merge1->initialize(panoWidth, panoHeight);
	m_merge2 = new GLSLMergeTexture();
	m_merge2->setGL(m_gl, m_functions_2_0);
	m_merge2->initialize(panoWidth, panoHeight);
	clear();
	
	m_initialized = true;
}

void GLSLPyramidSum::clear()
{
	lastTexture = -1;
	m_currentMerge = 1;
	m_merge2->clear();
}

void GLSLPyramidSum::render(GPUResourceHandle texture)
{
	if (lastTexture == -1)
	{
		m_merge1->render(m_merge2->getTargetGPUResource(), texture);
		lastTexture = m_merge1->getTargetGPUResource();
		m_currentMerge = 2;
	}
	else
	{
		if (m_currentMerge == 1)
		{
			m_merge1->render(lastTexture, texture);
			lastTexture = m_merge1->getTargetGPUResource();
			m_currentMerge = 2;
		}
		else
		{
			m_merge2->render(lastTexture, texture);
			lastTexture = m_merge2->getTargetGPUResource();
			m_currentMerge = 1;
		}
	}
}

const int GLSLPyramidSum::getWidth()
{
	return panoramaWidth;
}

const int GLSLPyramidSum::getHeight()
{
	return panoramaHeight;
}

GPUResourceHandle GLSLPyramidSum::getTargetGPUResource()
{
	return lastTexture;
}