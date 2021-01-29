#include "GLSLBanner.h"
#include "define.h"



// GLSLBill
GLSLBill::GLSLBill(QObject *parent) : GPUBill(parent)
{
}

GLSLBill::~GLSLBill()
{
}

void GLSLBill::initialize(int panoWidth, int panoHeight)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;

	// frame buffer
	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
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
	ADD_SHADER_FROM_CODE(m_program, "frag", "bill");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/bill.frag");
#endif
	m_program->link();
	m_program->bind();

	m_program->setUniformValue("bgTexture", 0);
	m_program->setUniformValue("bannerTexture", 1);
	m_program->setUniformValue("widthPixels", panoWidth);
	m_program->setUniformValue("heightPixels", panoHeight);
	
	m_program->release();

	m_initialized = true;
}

void GLSLBill::render(std::vector<BannerInfo*> bannerInputs)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = panoramaWidth;
	float height = panoramaHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);

	for (int i = 0; i < bannerInputs.size(); i++)
	{
		BannerInfo* pBanner = bannerInputs[i];

		m_gl->glActiveTexture(GL_TEXTURE1);
		m_gl->glBindTexture(GL_TEXTURE_2D, pBanner->billColorCvt->getTargetGPUResource());
		// homography
		m_program->setUniformValue("bannerPaiPlane", pBanner->paiPlane.mat_44);
		m_program->setUniformValue("bannerPaiZdotOrg", pBanner->paiZdotOrg);
		m_program->setUniformValue("bannerHomography", pBanner->homography.mat_33);

		m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

#if DEBUG_SHADER
		QString strSaveName = QString("Bill") + ".png";
		saveTexture(m_fboId, panoramaWidth, panoramaHeight, strSaveName, m_gl, false);
#endif 
	}

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();
}


// GLSLBanner
GLSLBanner::GLSLBanner(QObject *parent) : GPUBanner(parent)
{
}

GLSLBanner::~GLSLBanner()
{
}

void GLSLBanner::initialize(int panoWidth, int panoHeight)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;
	
	// frame buffer
	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
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
	ADD_SHADER_FROM_CODE(m_program, "frag", "banner");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/banner.frag");
#endif
	m_program->link();
	m_program->bind();
	
	m_program->setUniformValue("bgTexture", 0);
	m_program->setUniformValue("bannerTexture", 1);

	m_program->release();

	m_initialized = true;
}

void GLSLBanner::render(GPUResourceHandle srcTextureId, GPUResourceHandle billTextureId)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = panoramaWidth;
	float height = panoramaHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, srcTextureId);
	m_gl->glActiveTexture(GL_TEXTURE1);
	m_gl->glBindTexture(GL_TEXTURE_2D, billTextureId);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();
}
