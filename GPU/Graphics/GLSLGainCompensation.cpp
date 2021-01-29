#include "GLSLGainCompensation.h"
#include "common.h"
#include "define.h"

GLSLGainCompensation::GLSLGainCompensation(QObject *parent) : GPUGainCompensation(parent)
{
}

GLSLGainCompensation::~GLSLGainCompensation()
{
	if (m_initialized)
	{
		m_gl->glDeleteTextures(m_targetCount, m_fboTextureIds);
		m_gl->glDeleteFramebuffers(m_targetCount, m_fboIds);

		for (int i = 0; i < 2; i++)
		{
			m_pbos[i]->destroy();
			delete m_pbos[i];
			m_pbos[i] = NULL;
		}
	}
}

void GLSLGainCompensation::initialize(int cameraWidth, int cameraHeight)
{
	this->cameraWidth = cameraWidth;
	this->cameraHeight = cameraHeight;
	
	// frame buffer
	m_gl->glGenTextures(m_targetCount, m_fboTextureIds);
	m_gl->glGenFramebuffers(m_targetCount, m_fboIds);

	for (int i = 0; i < m_targetCount; i++)
	{
		m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureIds[i]);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, getWidth(), getHeight(), 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

		// load textures and create framebuffers
		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[i]);
		m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureIds[i], 0);
	}
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
	m_workingGPUResource = 0;

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "gainCompensation");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/gainCompensation.frag");
#endif
	m_program->link();
	m_program->bind();

	m_program->setUniformValue("texture", 0);
	m_gainUnif = m_program->uniformLocation("gain");

	for (int i = 0; i < 2; i++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelPackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::StreamRead);
		pbo->bind();
		pbo->allocate(cameraWidth * cameraHeight * 3);
		pbo->release();
		m_pbos[i] = pbo;
	}
	m_targetArrayIndex = 0;

	m_initialized = true;
}


void GLSLGainCompensation::render(GPUResourceHandle textureId, float ev)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[m_workingGPUResource]);

	float width = cameraWidth;
	float height = cameraHeight;

	float gain = ev2gain(ev);
	m_program->setUniformValue(m_gainUnif, gain);

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, textureId);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_workingGPUResource = (m_workingGPUResource + 1) % m_targetCount;

	m_program->release();
}

void GLSLGainCompensation::getRGBBuffer(unsigned char* rgbBuffer)
{
	if (rgbBuffer == NULL)
	{
		return;
	}
	if (m_initialized) {
		m_program->bind();
		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[getReadyGPUResourceIndex()]);

		m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
		QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
		QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_targetCount];

		pboCurrent->bind();
		m_functions_2_0->glReadPixels(0, 0, getWidth(), getHeight(), GL_BGR, GL_UNSIGNED_BYTE, 0);
		pboCurrent->release();

		pboNext->bind();
		GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::ReadOnly);
		if (ptr)
		{
			memcpy(rgbBuffer, ptr, getWidth() * getHeight() * 3);
			pboNext->unmap();
		}
		pboNext->release();

		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
		m_program->release();
	}
}


