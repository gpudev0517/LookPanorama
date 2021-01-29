#include "GLSLPanoramaPostProcessing.h"
#include "define.h"

GLSLPanoramaPostProcessing::GLSLPanoramaPostProcessing(QObject *parent) : GPUPanoramaPostProcessing(parent)
{
}

GLSLPanoramaPostProcessing::~GLSLPanoramaPostProcessing()
{
	m_gl->glDeleteTextures(1, &m_lutTexture);
	m_lutPBO->destroy();
	delete m_lutPBO;
}

void GLSLPanoramaPostProcessing::initialize(int panoWidth, int panoHeight)
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
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, panoramaWidth, panoramaHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//Lut

	m_gl->glGenTextures(1, &m_lutTexture);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_lutTexture);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	

	m_lutPBO = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
	m_lutPBO->create();
	m_lutPBO->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
	m_lutPBO->bind();
	m_lutPBO->allocate(256 * 4 * sizeof(unsigned char));
	m_lutPBO->release();

	m_lutPBO->bind();
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	m_lutPBO->release();
	GLenum error = glGetError();




	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "panoPostProcessing");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/panoColorCorrection.frag");
#endif
	m_program->link();
	m_program->bind();

	m_program->setUniformValue("colorTexture", 0);
	m_program->setUniformValue("lutTexture", 1);
	m_program->setUniformValue("seamTexture", 2);

	placeMatUnif = m_program->uniformLocation("placeMat");

	m_program->release();

	m_initialized = true;
}

void GLSLPanoramaPostProcessing::setLutData(QVariantList *vList){

	m_gl->glActiveTexture(GL_TEXTURE1);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_lutTexture);

	m_lutPBO->bind();

	float splitUnit = 255.f / (float)(LUT_COUNT-1);

	GLubyte *ptr = (GLubyte *)m_lutPBO->map(QOpenGLBuffer::WriteOnly);
	for (int k = 0; k < 4; k++){
		int s_j = 0;
		for (int i = 0; i < 256; i++){
			for (int j = 0; j < vList[k].size() - 1; j++){
				if (i >= j * splitUnit && i < (j + 1) * splitUnit){
					s_j = j;
					break;
				}
			}
			float alpha = (i - s_j * splitUnit) / splitUnit;
			float sample = (1.0f - alpha) * vList[k][s_j].toFloat() + alpha * vList[k][s_j + 1].toFloat();
			ptr[4 * i + k] = fmin(sample * 255.f, 255.f);
		}
	}
	m_lutPBO->unmap();

	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	m_lutPBO->release();

}


void GLSLPanoramaPostProcessing::render(GLuint panoTextureId, vec3 ctLightColor,
	float yaw, float pitch, float roll,
	int seamTextureId)
{
	mat3 m = mat3_id, invM = mat3_id;
	vec3 u(roll * sd_to_rad, pitch * sd_to_rad, yaw * sd_to_rad);
	m.set_rot_zxy(u);
	invert(invM, m);

	m_program->bind();

	QMatrix3x3 matrix;
	float* data = matrix.data();
	memcpy(data, (float*)invM.mat_array, sizeof(float) * 9);
	m_program->setUniformValue(placeMatUnif, matrix);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = getWidth();
	float height = getHeight();

	m_program->setUniformValue("ctLightColor", ctLightColor.x, ctLightColor.y, ctLightColor.z);

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, panoTextureId);

	m_gl->glActiveTexture(GL_TEXTURE1);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_lutTexture);

	if (seamTextureId == -1)
	{
		m_program->setUniformValue("seamOn", false);
	}
	else
	{
		m_program->setUniformValue("seamOn", true);
		m_gl->glActiveTexture(GL_TEXTURE2);
		m_gl->glBindTexture(GL_TEXTURE_2D, seamTextureId);
	}
	
	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();
}
