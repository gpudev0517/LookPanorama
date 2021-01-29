#include "GLSLNodalBlending.h"
#include "define.h"
#include "common.h"

GLSLNodalBlending::GLSLNodalBlending(QObject *parent) : GPUNodalBlending(parent)
{
}

GLSLNodalBlending::~GLSLNodalBlending()
{
}

void GLSLNodalBlending::initialize(int panoWidth, int panoHeight, int nodalCameraCount, bool haveNodalMaskImage)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;
	this->nodalCameraCount = nodalCameraCount;
	this->haveNodalMaskImage = haveNodalMaskImage;
	
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
	ADD_SHADER_FROM_CODE(m_program, "frag", "nodalBlend");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/composite.frag");
#endif
	m_program->link();
	m_program->bind();

	m_program->setUniformValue("foregroundTexture", 0);
	int colorTextureIds[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	m_gl->glUniform1iv(m_program->uniformLocation(QString("bgTexture")), 8, colorTextureIds);

	int weightTextureIds[8] = { 9, 10, 11, 12, 13, 14, 15, 16 };
	m_gl->glUniform1iv(m_program->uniformLocation(QString("bgWeightTexture")), 8, weightTextureIds);

	m_program->setUniformValue("bgCnt", nodalCameraCount);

	if (haveNodalMaskImage)
		m_program->setUniformValue("nodalWeightOn", 1);	
	else
		m_program->setUniformValue("nodalWeightOn", 0);
	

	m_program->release();

	m_initialized = true;
}


void GLSLNodalBlending::render(GPUResourceHandle fbo1, QList<int> nodalTextures, QList<int> nodalWeightTextures)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = panoramaWidth;
	float height = panoramaHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, fbo1);

	GLuint bgWeightOn[8];

	for (int i = 0; i < nodalCameraCount; i++) {
		m_gl->glActiveTexture(GL_TEXTURE1 + i);
		m_gl->glBindTexture(GL_TEXTURE_2D, nodalTextures[i]);

		if (nodalWeightTextures[i] == -1)
		{
			bgWeightOn[i] = false;
		}
		else
		{
			bgWeightOn[i] = true;
			m_gl->glActiveTexture(GL_TEXTURE9 + i);
			m_gl->glBindTexture(GL_TEXTURE_2D, nodalWeightTextures[i]);
		}
	}
	m_program->setUniformValueArray("nodalWeightOn", (GLuint*)bgWeightOn, nodalCameraCount);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();

#if DEBUG_SHADER
	QString strSaveName = QString("feathering_") + QString::number(compositeID) + ".png";
	saveTexture(m_fboId, width, height, strSaveName, m_gl);
#endif 
}