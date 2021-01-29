#include "GLSLBlending.h"
#include "define.h"

GLSLBlending::GLSLBlending(QObject *parent) : GPUProgram(parent)
{
}

GLSLBlending::~GLSLBlending()
{
}

void GLSLBlending::initialize(int panoWidth, int panoHeight, int viewCount)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;
	this->m_viewCount = viewCount;
	
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
	ADD_SHADER_FROM_CODE(m_program, "frag", "blending");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/blending.frag");
#endif
	m_program->link();
	m_program->bind();

	int colorTextureIds[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	m_gl->glUniform1iv(m_program->uniformLocation(QString("colorMap")), 8, colorTextureIds);
	int alphaTextureIds[8] = { 8, 9, 10, 11, 12, 13, 14, 15 };
	m_gl->glUniform1iv(m_program->uniformLocation(QString("blendMap")), 8, alphaTextureIds);
	m_program->setUniformValue("viewCnt", m_viewCount);

	levelScaleUnif = m_program->uniformLocation("diffLevelScale");

	m_initialized = true;
}


void GLSLBlending::render(GPUResourceHandle colorTextures[], GPUResourceHandle alphaTextures[], int colorLevel, int alphaLevel, int alphaType)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = panoramaWidth;
	float height = panoramaHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	for (int i = 0; i < m_viewCount; i++)
	{
		m_gl->glActiveTexture(GL_TEXTURE0 + i);
		m_gl->glBindTexture(GL_TEXTURE_2D, colorTextures[i]);

		m_gl->glActiveTexture(GL_TEXTURE8 + i);
		m_gl->glBindTexture(GL_TEXTURE_2D, alphaTextures[i]);
	}

	float levelScale = getLevelScale(colorLevel);
	m_program->setUniformValue(levelScaleUnif, getLevelScale(colorLevel - alphaLevel));
	m_program->setUniformValue("levelScale", levelScale);
	m_program->setUniformValue("alphaType", alphaType);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();
}

const int GLSLBlending::getWidth()
{
	return panoramaWidth;
}

const int GLSLBlending::getHeight()
{
	return panoramaHeight;
}

float GLSLBlending::getLevelScale(int level)
{
	int scale = 1;
	for (int i = 0; i < level; i++)
		scale *= 2;
	return scale;
}