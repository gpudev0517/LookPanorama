#include "GLSLPanoramaInteract.h"
#include "3DMath.h"
#include "define.h"
#include "OculusViewer.h"

#define HORZ_SEGS 256
#define VERT_SEGS 128

#define TEST	0

GLSLPanoramaInteract::GLSLPanoramaInteract(QObject *parent) : GPUProgram(parent)
{
	vertices = NULL;
	texCoords = NULL;
}

GLSLPanoramaInteract::~GLSLPanoramaInteract()
{
	clear();
}

void GLSLPanoramaInteract::clear()
{
	panoramaWidth = 0;
	panoramaHeight = 0;
	m_projMat = mat4_id;
	m_viewMat = mat4_id;

	if (m_initialized)
	{
		if (vertices)
		{
			delete[] vertices;
			vertices = NULL;
		}
			
		if (texCoords)
		{
			delete[] texCoords;
			texCoords = NULL;
		}
			
		/*m_gl->glDeleteRenderbuffers(1, &m_renderbuffer);
		m_renderbuffer = 0;*/
	}
}

void GLSLPanoramaInteract::initialize(int panoWidth, int panoHeight)
{
	clear();

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

	// Create a depth buffer for our framebuffer
	m_gl->glGenRenderbuffers(1, &m_renderbuffer);
	m_gl->glBindRenderbuffer(GL_RENDERBUFFER, m_renderbuffer);
	m_gl->glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, panoramaWidth, panoramaHeight);

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_renderbuffer);
	m_gl->glEnable(GL_DEPTH_TEST);
	m_gl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "Oculus");
	ADD_SHADER_FROM_CODE(m_program, "frag", "Interact");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/Oculus.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/Interact.frag");
#endif
	m_program->link();
	m_vertexAttr = m_program->attributeLocation("vertex");
	m_texCoordAttr = m_program->attributeLocation("texCoord");
	
	m_program->bind();
	m_program->setUniformValue("texture", 0);
	m_program->release();

	m_modelViewMatrix = m_program->uniformLocation("matrix");

	// Create a unit sphere
	float PI = 3.1415927;
	vertices = new GLfloat[VERT_SEGS*HORZ_SEGS * 6 * 3];
	texCoords = new GLfloat[VERT_SEGS*HORZ_SEGS * 6 * 2];
	for (int isS = 0; isS < VERT_SEGS; isS++)
	{
		float y1 = 1.0f * isS / (float) VERT_SEGS;
		float y2 = 1.0f * (isS + 1) / (float) VERT_SEGS;
		float phi1 = (-1.0f + 2.0f * y1) * PI / 2.0f;
		float phi2 = (-1.0f + 2.0f * y2) * PI / 2.0f;	
		for (int j = 0; j < HORZ_SEGS; j++)
		{
			float x1 = 1.0f * j / HORZ_SEGS;
			float x2 = 1.0f * (j + 1) / HORZ_SEGS;

			float theta1 = (2.0f * x1 - 1.0f) * PI;
			float theta2 = (2.0f * x2 - 1.0f) * PI;

			vec3 p11 = vec3(cosf(phi1) * sinf(theta1), sinf(phi1), -cosf(phi1) * cosf(theta1));
			vec3 p12 = vec3(cosf(phi1) * sinf(theta2), sinf(phi1), -cosf(phi1) * cosf(theta2));
			vec3 p21 = vec3(cosf(phi2) * sinf(theta1), sinf(phi2), -cosf(phi2) * cosf(theta1));
			vec3 p22 = vec3(cosf(phi2) * sinf(theta2), sinf(phi2), -cosf(phi2) * cosf(theta2));
			vec2 t11 = vec2(x1, y1);
			vec2 t12 = vec2(x2, y1);
			vec2 t21 = vec2(x1, y2);
			vec2 t22 = vec2(x2, y2);

			// 11 - 12
			// |  /
			// 21
			for (int k = 0; k < 3; k++)
			{
				vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 0 * 3 + k] = p11[k];
				vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 1 * 3 + k] = p12[k];
				vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 2 * 3 + k] = p21[k];
			}
			for (int k = 0; k < 2; k++)
			{
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 0 * 2 + k] = t11[k];
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 1 * 2 + k] = t12[k];
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 2 * 2 + k] = t21[k];
			}

			// 21 - 12
			//    /
			// 22
			for (int k = 0; k < 3; k++)
			{
				vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 3 * 3 + k] = p21[k];
				vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 4 * 3 + k] = p12[k];
				vertices[(isS * HORZ_SEGS + j) * 6 * 3 + 5 * 3 + k] = p22[k];
			}
			for (int k = 0; k < 2; k++)
			{
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 3 * 2 + k] = t21[k];
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 4 * 2 + k] = t12[k];
				texCoords[(isS * HORZ_SEGS + j) * 6 * 2 + 5 * 2 + k] = t22[k];
			}
		}
	}
	m_initialized = true;
}

void GLSLPanoramaInteract::render(GPUResourceHandle originalTextures, bool isStereo, float yaw, float pitch, float roll)
{
	if (!m_program)
		return;
	vec3 u(roll * sd_to_rad, pitch * sd_to_rad, yaw * sd_to_rad);
	mat3 m = mat3_id;
	m.set_rot_zxy(u);

	m_program->bind();
	
	float aspectRatio = panoramaWidth / (float)panoramaHeight;

	mat4 view = mat4_id;
	mat4 matrix = mat4_id;
#if 0
	vec3 eyePos = vec3_null;
	vec3 up = vec3_y;
	vec3 target = vec3_z;
	eyePos.z = 0;
	target = eyePos + vec3_neg_z;
	look_at(view, eyePos, target, up);
#endif
	view.set_mat3(m);
	m_viewMat = view;
	perspective(m_projMat, 60, aspectRatio, 0.001f, 2.f);
	mult(matrix, m_projMat, m_viewMat);

	QMatrix4x4 finalMat;
	float* data = finalMat.data();
	memcpy(data, (float*)matrix.mat_array, sizeof(float)* 16);

	m_program->setUniformValue(m_modelViewMatrix, finalMat);

	m_gl->glEnable(GL_DEPTH_TEST);
	//m_gl->glEnable(GL_CLIP_DISTANCE0);

	m_gl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fboId);

	float width = panoramaWidth;
	float height = panoramaHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, originalTextures);

	m_gl->glEnableVertexAttribArray(m_vertexAttr);
	m_gl->glEnableVertexAttribArray(m_texCoordAttr);

	m_gl->glVertexAttribPointer(m_vertexAttr, 3, GL_FLOAT, GL_FALSE, 0, vertices);
	m_gl->glVertexAttribPointer(m_texCoordAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords);

	if (isStereo)
	{
		m_program->setUniformValue("panoSection", 0);
		m_gl->glViewport(0, 0, width / 2, height);
		m_gl->glDrawArrays(GL_TRIANGLES, 0, HORZ_SEGS * VERT_SEGS * 6);

		m_program->setUniformValue("panoSection", 1);
		m_gl->glViewport(width / 2, 0, width / 2, height);
		m_gl->glDrawArrays(GL_TRIANGLES, 0, HORZ_SEGS * VERT_SEGS * 6);
	}
	else
	{
		m_program->setUniformValue("panoSection", 2);
		m_gl->glViewport(0, 0, width, height);
		m_gl->glDrawArrays(GL_TRIANGLES, 0, HORZ_SEGS * VERT_SEGS * 6);
	}

	m_gl->glDisableVertexAttribArray(m_texCoordAttr);
	m_gl->glDisableVertexAttribArray(m_vertexAttr);

	m_gl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	m_gl->glDisable(GL_DEPTH_TEST);

	m_program->release();
}

const int GLSLPanoramaInteract::getWidth()
{
	return panoramaWidth;
}

const int GLSLPanoramaInteract::getHeight()
{
	return panoramaHeight;
}