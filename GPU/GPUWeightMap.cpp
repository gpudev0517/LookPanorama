#include "GPUWeightMap.h"
#include "define.h"
#include "common.h"
#include "define.h"

#define DELTA_SCALE 3
#define BRUSH_X_RES 500
#define BRUSH_Y_RES 250

GLfloat* GPUDeltaWeightMap::vertices;

//Camera Weight Map
GPUCameraWeightMap::GPUCameraWeightMap(QObject *parent) : GPUProgram(parent)
{
}

GPUCameraWeightMap::~GPUCameraWeightMap()
{
}

//Panorama Weight Map


GPUPanoramaWeightMap::GPUPanoramaWeightMap(QObject *parent, bool isYUV) : GPUProgram(parent)
{
}

GPUPanoramaWeightMap::~GPUPanoramaWeightMap()
{
}

//For Delta Weight Map

GPUDeltaWeightMap::GPUDeltaWeightMap(QObject *parent, bool isYUV) : GPUProgram(parent)
, m_Name("GPUDeltaWeightMap")
//, m_srcTexture(0)
{
}

GPUDeltaWeightMap::~GPUDeltaWeightMap()
{
	if (m_initialized)
	{
		m_functions_4_3->glDeleteBuffers(1, &ubo);
	}
}

void GPUDeltaWeightMap::initialize(int xres, int yres, int panoWidth, int panoHeight)
{
	panoramaWidth = panoWidth;
	panoramaHeight = panoHeight;

	camWidth = xres;
	camHeight = yres;

	// frame buffer
	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, camWidth, camHeight, 0, GL_RED, GL_FLOAT, NULL);

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "deltaWeight");
	ADD_SHADER_FROM_CODE(m_program, "frag", "deltaWeight");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/deltaWeight.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/deltaWeight.frag");
#endif
	m_program->link();

	m_program->bind();

	m_vertexAttr = m_program->attributeLocation("vertex");
	m_texCoordAttr = m_program->attributeLocation("texCoord");

	QMatrix4x4 matrix;
	matrix.ortho(-camWidth / 2, camWidth / 2, -camHeight / 2, camHeight / 2, -10.0f, 10.0f);
	matrix.translate(0, 0, -2);
	m_program->setUniformValue("matrix", matrix);

	m_program->setUniformValue("panoramaWidth", (float)panoramaWidth);
	m_program->setUniformValue("panoramaHeight", (float)panoramaHeight);

	m_functions_4_3->glGenBuffers(1, &ubo);

	m_program->release();

	if (vertices == 0)
	{
		int ww = BRUSH_X_RES;
		int hh = BRUSH_Y_RES;

		vertices = new GLfloat[ww * hh * 8];
		int idx = 0;
		for (int j = 0; j < hh; j++){
			for (int i = 0; i < ww; i++){
				vertices[idx++] = i / (float)ww;
				vertices[idx++] = j / (float)hh;

				vertices[idx++] = i / (float)ww;
				vertices[idx++] = (j + 1) / (float)hh;

				vertices[idx++] = (i + 1) / (float)ww;
				vertices[idx++] = (j + 1) / (float)hh;

				vertices[idx++] = (i + 1) / (float)ww;
				vertices[idx++] = j / (float)hh;
			}
		}
	}

	resetMap();

	m_initialized = true;
}

void GPUDeltaWeightMap::saveWeightmap(QString fileName)
{
	int imageW = camWidth;
	int imageH = camHeight;

	glBindTexture(GL_TEXTURE_2D, getTargetGPUResource());

	uchar* dstBuffer = new uchar[imageW * imageH];
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, dstBuffer);

	QImage image((uchar*)dstBuffer, imageW, imageH, QImage::Format::Format_Grayscale8);
	image.save(fileName, NULL, 100);
	delete[] dstBuffer;
}

bool GPUDeltaWeightMap::loadWeightmap(QString strFilename)
{
	QImage image;
	
	if (!image.load(strFilename, "png"))
	{
		PANO_N_LOG(QString("WeightMap (%1) does NOT exist").arg(strFilename));
		return false;
	}

	if (camWidth != image.width() || camHeight != image.height())
	{
		PANO_N_LOG(QString("WeightMap information is NOT same with camera.(%1)").arg(strFilename));
		return false;
	}

	int w = camWidth;
	int h = camHeight;

	PANO_LOG(QString(">WeightMap (%1) file loaded successfully").arg(strFilename));

	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camWidth, camHeight, GL_RED, GL_UNSIGNED_BYTE, image.constBits());

	return true;
}

void GPUDeltaWeightMap::updateCameraParams()
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

void GPUDeltaWeightMap::render(float radius, float falloff, float strength, float centerx, float centery, bool increment, mat3 &globalM, int camID)
{
#if 0
	QMatrix3x3 mYaw = getViewMatrix(setting.m_fYaw, 0, 0);
	QMatrix3x3 mPitch = getViewMatrix(0, setting.m_fPitch, 0);
	QMatrix3x3 mRoll = getViewMatrix(0, 0, setting.m_fRoll);
	QMatrix3x3 m2 = getViewMatrix(cam.m_yaw, cam.m_pitch, cam.m_roll);
	QMatrix3x3 m = m2 * mYaw * mPitch * mRoll;
	m_program->setUniformValue(cpUnif, m);
#else
	cam.cP.set_mat3(globalM);
#endif
	updateCameraParams();

	m_program->bind();

	if (!increment)
		strength *= -1;

	m_program->setUniformValue("radius", radius);
	m_program->setUniformValue("falloff", falloff);
	m_program->setUniformValue("strength", strength);
	m_program->setUniformValue("centerx", centerx);
	m_program->setUniformValue("centery", centery);

	GLenum error = glGetError();

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	m_gl->glViewport(0, 0, camWidth, camHeight);

	GLuint uboIndex = m_functions_4_3->glGetUniformBlockIndex(m_program->programId(), "cameraBuffer"); // get index of block
	m_functions_4_3->glBindBuffer(GL_UNIFORM_BUFFER, ubo);
	m_functions_4_3->glBufferData(GL_UNIFORM_BUFFER, sizeof(CameraData),
		&cam, GL_STATIC_DRAW);
	m_functions_4_3->glBindBufferBase(GL_UNIFORM_BUFFER, uboIndex, ubo);
	int vertexAttr = m_vertexAttr;

	m_gl->glVertexAttribPointer(vertexAttr, 2, GL_FLOAT, GL_FALSE, 0, vertices);

	m_gl->glEnableVertexAttribArray(vertexAttr);

	int ww = BRUSH_X_RES;
	int hh = BRUSH_Y_RES;
	m_gl->glDrawArrays(GL_QUADS, 0, 4 * ww * hh);
	m_gl->glFinish();

	error = glGetError();

	m_gl->glDisableVertexAttribArray(vertexAttr);

	glDisable(GL_BLEND);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();

#if DEBUG_SHADER
	QString strSaveName = QString("deltaweight_") + QString::number(camID) + ".png";
	saveTexture(m_fboId, camWidth, camHeight, strSaveName, m_gl);
#endif 
}

void GPUDeltaWeightMap::resetMap()
{
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glClearColor(0.5f, 0, 0, 1.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
}