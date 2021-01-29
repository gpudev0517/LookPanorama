#include "GLSLBoxBlur.h"
#include "define.h"
#include "common.h"

GLSLBoxBlur::GLSLBoxBlur(QObject *parent) : GPUProgram(parent)
{
}

GLSLBoxBlur::~GLSLBoxBlur()
{
}

void GLSLBoxBlur::initialize(int panoWidth, int panoHeight, bool isVertical, bool isPartial)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;
	this->isPartial = isPartial;
	
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
	ADD_SHADER_FROM_CODE(m_program, "frag", "boxBlur");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/boxBlur.frag");
#endif
	m_program->link();
	m_program->bind();

	m_program->setUniformValue("texture", 0);
	m_program->setUniformValue("width", panoramaWidth);
	m_program->setUniformValue("height", panoramaHeight);
	m_program->setUniformValue("isVertical", isVertical);

	blurRadiusUnif = m_program->uniformLocation("blurRadius");
	isPartialUnif = m_program->uniformLocation("isPartial");

	m_program->release();

	m_initialized = true;
}

void GLSLBoxBlur::render(GPUResourceHandle textureId, int blurRadius, float alphaType)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = panoramaWidth;
	float height = panoramaHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, textureId);

	m_program->setUniformValue(blurRadiusUnif, blurRadius);
	m_program->setUniformValue(isPartialUnif, isPartial);
	m_program->setUniformValue("alphaType", alphaType);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();
}

const int GLSLBoxBlur::getWidth()
{
	return panoramaWidth;
}

const int GLSLBoxBlur::getHeight()
{
	return panoramaHeight;
}