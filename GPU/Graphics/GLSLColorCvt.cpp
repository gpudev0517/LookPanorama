#include "GLSLColorCvt.h"
#include "common.h"


GLSLColorCvt_2RGBA::GLSLColorCvt_2RGBA(QObject *parent) : GPUColorCvt_2RGBA(parent)
{
}

GLSLColorCvt_2RGBA::~GLSLColorCvt_2RGBA()
{
	if (m_initialized)
	{
		m_gl->glDeleteTextures(m_targetCount, m_fboTextureIds);
		m_gl->glDeleteFramebuffers(m_targetCount, m_fboIds);

		m_gl->glDeleteTextures(1, &m_srcTextureId);

		for (int i = 0; i < m_pbos.size(); i++)
		{
			m_pbos[i]->destroy();
			delete m_pbos[i];
		}
		m_pbos.clear();
	}
}


void GLSLColorCvt_2RGBA::initialize(int imgWidth, int imgHeight)
{
	imageWidth = 0;
	imageHeight = 0;
	imageBytesPerLine = 0;

	m_gl->glGenTextures(m_targetCount, m_fboTextureIds);
	m_gl->glGenFramebuffers(m_targetCount, m_fboIds);

	for (int i = 0; i < m_targetCount; i++)
	{
		m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureIds[i]);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imgWidth, imgHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

		// load textures and create framebuffers
		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[i]);
		m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureIds[i], 0);
	}
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
	m_workingGPUResource = 0;

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
	addShaderFile();
	bool suc = m_program->link();

	m_program->bind();


	m_program->setUniformValue("texture", 0);
	m_program->setUniformValue("textureU", 1);
	m_program->setUniformValue("textureV", 2);

	bytesPerLineUnif = m_program->uniformLocation("bytesPerLine");
	imageWidthUnif = m_program->uniformLocation("width");
	imageHeightUnif = m_program->uniformLocation("height");


	//addPBO();

	m_targetArrayIndex = 0;

	m_program->release();

	m_initialized = true;

	renderEmptyFrame();
}


void GLSLColorCvt_2RGBA::render(ImageBufferData& img)
{
	//qDebug("GLSLColorCvt_2RGB - 1");
	m_program->bind();

	// Simple glTexImage2D, with PBO
	//const uchar * bits = img.constBits();
	int width = img.mImageY.width;
	int height = img.mImageY.height;

	if (width == 0 || height == 0) return;

	bool blTextureFormatChanged = false;
	if (imageWidth != width || imageHeight != height)
	{
		imageWidth = width;
		imageHeight = height;
		imageBytesPerLine = img.mImageY.stride;

		if (isColorConversionNeeded())
		{
			m_program->setUniformValue(bytesPerLineUnif, imageBytesPerLine);
			m_program->setUniformValue(imageWidthUnif, imageWidth);
			m_program->setUniformValue(imageHeightUnif, imageHeight);
		}

		createSourceGPUResource();
		
		addPBO();
		blTextureFormatChanged = true;
	}


	const unsigned char * buffers[3] = { img.mImageY.buffer, img.mImageU.buffer, img.mImageV.buffer };
	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	//qDebug("GLSLColorCvt_2RGB - 3");
	m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
	for (int i = 0; i < 3; i++)
	{
		// host to device buffer uploading.
		setSourceGPUResourceBuffer(i, buffers[i], blTextureFormatChanged);
	}
	//qDebug("GLSLColorCvt_2RGB - 4");
	// Don't comment below code block absolutely.
	if (blTextureFormatChanged)
	{	// It's very important to avoid empty-rendering.
		m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
		for (int i = 0; i < 3; i++)
		{	// host to device buffer uploading.
			setSourceGPUResourceBuffer(i, buffers[i], false);

		}
	}

	//qDebug("GLSLColorCvt_2RGB - 2");
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[m_workingGPUResource]);

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, imageWidth, imageHeight);

	m_program->setUniformValue("mirrorVert", true);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
	

	m_workingGPUResource = (m_workingGPUResource + 1) % m_targetCount;

	m_program->release();
	//qDebug("GLSLColorCvt_2RGB - 6");
}

//////////////

GLSLColorCvt_YUV2RGBA::GLSLColorCvt_YUV2RGBA(QObject *parent) : GLSLColorCvt_2RGBA(parent)
{
}

GLSLColorCvt_YUV2RGBA::~GLSLColorCvt_YUV2RGBA()
{

}

void GLSLColorCvt_YUV2RGBA::createSourceGPUResource()
{
	// Source texture
	m_gl->glGenTextures(1, &m_srcTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageBytesPerLine, imageHeight * 3 / 2, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
}

void GLSLColorCvt_YUV2RGBA::addShaderFile()
{
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "yuv4202rgb");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/yuv4202rgb.frag");
#endif
}

void GLSLColorCvt_YUV2RGBA::addPBO()
{
	for (int i = 0; i < m_targetCount; i++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
		pbo->bind();
		pbo->allocate(imageBytesPerLine * imageHeight * 3 / 2);
		pbo->release();
		m_pbos.push_back(pbo);
	}
}

void GLSLColorCvt_YUV2RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	//qDebug("setSourceGPUResourceBuffer - 1");
	buffer_ptrs[index] = bits;

	if (index == 2)
	{
		//qDebug() << m_pbos.size();
		//qDebug() << m_targetArrayIndex;
		int widths[3] = { imageBytesPerLine, imageBytesPerLine / 2, imageBytesPerLine / 2 };
		int heights[3] = { imageHeight, imageHeight / 2, imageHeight / 2 };
		QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
		QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_targetCount];
		//qDebug("setSourceGPUResourceBuffer - 2");
		pboCurrent->bind();
		if (textureChanged)
			m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageBytesPerLine, imageHeight * 3 / 2, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
		else
			m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageBytesPerLine, imageHeight * 3 / 2, GL_RED, GL_UNSIGNED_BYTE, 0);
		pboCurrent->release();
		//qDebug("setSourceGPUResourceBuffer - 3");
		if (imageBytesPerLine > 0 && imageHeight > 0)
		{
			pboNext->bind();
			GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::WriteOnly);
			if (ptr)
			{
				int offset = 0;
				for (int i = 0; i < 3; i++)
				{
					int len = widths[i] * heights[i];
					memcpy(ptr + offset, buffer_ptrs[i], len);
					offset += len;
				}
				pboNext->unmap();
			}
			pboNext->release();
		}
	}
}

bool GLSLColorCvt_YUV2RGBA::isColorConversionNeeded()
{
	return true;
}

//////////////

GLSLColorCvt_YUV4222RGBA::GLSLColorCvt_YUV4222RGBA(QObject *parent) : GLSLColorCvt_2RGBA(parent)
{
}

GLSLColorCvt_YUV4222RGBA::~GLSLColorCvt_YUV4222RGBA()
{

}

void GLSLColorCvt_YUV4222RGBA::createSourceGPUResource()
{
	// Source texture
	m_gl->glGenTextures(1, &m_srcTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageBytesPerLine / 4, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageBytesPerLine / 2, imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
}

void GLSLColorCvt_YUV4222RGBA::addShaderFile()
{
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "yuv4222rgb");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/yuv4222rgb.frag");
#endif
}

void GLSLColorCvt_YUV4222RGBA::addPBO()
{
	for (int j = 0; j < m_targetCount; j++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
		pbo->bind();
		pbo->allocate(imageBytesPerLine * imageHeight);
		pbo->release();
		m_pbos.push_back(pbo);
	}
}

void GLSLColorCvt_YUV4222RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	if (index != 0)
		return;

#if 1
	QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
	QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_targetCount];

	int widthPixels = imageBytesPerLine / 4;

	pboCurrent->bind();
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, widthPixels, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, widthPixels, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	pboCurrent->release();

	if (imageBytesPerLine > 0 && imageHeight > 0)
	{
		pboNext->bind();
		GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::WriteOnly);
		if (ptr)
		{
			memcpy(ptr, bits, imageBytesPerLine * imageHeight);
			pboNext->unmap();
		}
		pboNext->release();
	}
#else
	if (blTextureFormatChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height * 3 / 2, 0, GL_RED, GL_UNSIGNED_BYTE, bufferPtrs[0]);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height * 3 / 2, GL_RED, GL_UNSIGNED_BYTE, bufferPtrs[0]);
#endif
}

bool GLSLColorCvt_YUV4222RGBA::isColorConversionNeeded()
{
	return true;
}

//////////////

GLSLColorCvt_YUVJ422P2RGBA::GLSLColorCvt_YUVJ422P2RGBA(QObject *parent) : GLSLColorCvt_2RGBA(parent)
{
}

GLSLColorCvt_YUVJ422P2RGBA::~GLSLColorCvt_YUVJ422P2RGBA()
{

}

void GLSLColorCvt_YUVJ422P2RGBA::createSourceGPUResource()
{
	// Source texture
	m_gl->glGenTextures(1, &m_srcTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageBytesPerLine, imageHeight * 2, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
}

void GLSLColorCvt_YUVJ422P2RGBA::addShaderFile()
{
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "yuvj422p2rgb");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/yuv4222rgb.frag");
#endif
}

void GLSLColorCvt_YUVJ422P2RGBA::addPBO()
{
	for (int j = 0; j < m_targetCount; j++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
		pbo->bind();
		pbo->allocate(imageBytesPerLine * imageHeight * 2);
		pbo->release();
		m_pbos.push_back(pbo);
	}
}

void GLSLColorCvt_YUVJ422P2RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	buffer_ptrs[index] = bits;

#if 1
	if (index == 2)
	{
		int widths[3] = { imageBytesPerLine, imageBytesPerLine / 2, imageBytesPerLine / 2 };
		int heights[3] = { imageHeight, imageHeight, imageHeight };
		QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
		QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_targetCount];
		pboCurrent->bind();
		if (textureChanged)
			m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageBytesPerLine, imageHeight * 2, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
		else
			m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageBytesPerLine, imageHeight * 2, GL_RED, GL_UNSIGNED_BYTE, 0);
		pboCurrent->release();
		if (imageBytesPerLine > 0 && imageHeight > 0)
		{
			pboNext->bind();
			GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::WriteOnly);
			if (ptr)
			{
				int offset = 0;
				for (int i = 0; i < 3; i++)
				{
					int len = widths[i] * heights[i];
					memcpy(ptr + offset, buffer_ptrs[i], len);
					offset += len;
				}
				pboNext->unmap();
			}
			pboNext->release();
		}
	}
#else
	if (blTextureFormatChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height * 3 / 2, 0, GL_RED, GL_UNSIGNED_BYTE, bufferPtrs[0]);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height * 3 / 2, GL_RED, GL_UNSIGNED_BYTE, bufferPtrs[0]);
#endif
}

bool GLSLColorCvt_YUVJ422P2RGBA::isColorConversionNeeded()
{
	return true;
}

//////////////
GLSLColorCvt_BGR02RGBA::GLSLColorCvt_BGR02RGBA(QObject *parent) : GLSLColorCvt_2RGBA(parent)
{
}

GLSLColorCvt_BGR02RGBA::~GLSLColorCvt_BGR02RGBA()
{

}

void GLSLColorCvt_BGR02RGBA::createSourceGPUResource()
{
	// Source texture
	m_gl->glGenTextures(1, &m_srcTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void GLSLColorCvt_BGR02RGBA::addShaderFile()
{
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "bgr02rgb");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/rgb2rgb.frag");
#endif
}

void GLSLColorCvt_BGR02RGBA::addPBO()
{
	for (int i = 0; i < m_targetCount; i++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
		pbo->bind();
		pbo->allocate(imageWidth * imageHeight * 4);
		pbo->release();
		m_pbos.push_back(pbo);
	}
}

void GLSLColorCvt_BGR02RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	if (index != 0)
		return;
#if 1
	QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
	QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_targetCount];

	pboCurrent->bind();
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	pboCurrent->release();

	if (imageWidth > 0 && imageHeight > 0)
	{
		pboNext->bind();
		GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::WriteOnly);
		if (ptr)
		{
			memcpy(ptr, bits, imageHeight * imageBytesPerLine);
			pboNext->unmap();
		}
		pboNext->release();
	}
#else
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, bits);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGB, GL_UNSIGNED_BYTE, bits);
#endif
}

bool GLSLColorCvt_BGR02RGBA::isColorConversionNeeded()
{
	return false;
}


//////////////
GLSLColorCvt_BAYERRG82RGBA::GLSLColorCvt_BAYERRG82RGBA(QObject *parent) : GLSLColorCvt_2RGBA(parent)
{
}

GLSLColorCvt_BAYERRG82RGBA::~GLSLColorCvt_BAYERRG82RGBA()
{

}

void GLSLColorCvt_BAYERRG82RGBA::createSourceGPUResource()
{
	// Source texture
	m_gl->glGenTextures(1, &m_srcTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageWidth, imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
}

void GLSLColorCvt_BAYERRG82RGBA::addShaderFile()
{
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "bayerrg82rgb");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/bayerrg82rgb.frag");
#endif
}

void GLSLColorCvt_BAYERRG82RGBA::addPBO()
{
	for (int i = 0; i < m_targetCount; i++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
		pbo->bind();
		pbo->allocate(imageWidth * imageHeight);
		pbo->release();
		m_pbos.push_back(pbo);
	}
}

void GLSLColorCvt_BAYERRG82RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	if (index != 0)
		return;
#if 1
	QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
	QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_targetCount];

	pboCurrent->bind();
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageWidth, imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RED, GL_UNSIGNED_BYTE, 0);
	pboCurrent->release();

	if (imageWidth > 0 && imageHeight > 0)
	{
		pboNext->bind();
		GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::WriteOnly);
		if (ptr)
		{
			memcpy(ptr, bits, imageHeight * imageBytesPerLine);
			pboNext->unmap();
		}
		pboNext->release();
	}
#else
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, bits);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGB, GL_UNSIGNED_BYTE, bits);
#endif
}

bool GLSLColorCvt_BAYERRG82RGBA::isColorConversionNeeded()
{
	return true;
}




//////////////
GLSLColorCvt_RGB2RGBA::GLSLColorCvt_RGB2RGBA(QObject *parent) : GLSLColorCvt_2RGBA(parent)
{
}

GLSLColorCvt_RGB2RGBA::~GLSLColorCvt_RGB2RGBA()
{

}

void GLSLColorCvt_RGB2RGBA::createSourceGPUResource()
{
	// Source texture
	m_gl->glGenTextures(1, &m_srcTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
}

void GLSLColorCvt_RGB2RGBA::addShaderFile()
{
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "rgb2rgb");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/rgb2rgb.frag");
#endif
}

void GLSLColorCvt_RGB2RGBA::addPBO()
{
	for (int i = 0; i < m_targetCount; i++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
		pbo->bind();
		pbo->allocate(imageWidth * imageHeight * 3);
		pbo->release();
		m_pbos.push_back(pbo);
	}
}

void GLSLColorCvt_RGB2RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	if (index != 0)
		return;
#if 1
	QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
	QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_targetCount];

	pboCurrent->bind();
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGB, GL_UNSIGNED_BYTE, 0);
	pboCurrent->release();

	if (imageWidth > 0 && imageHeight > 0)
	{
		pboNext->bind();
		GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::WriteOnly);
		if (ptr)
		{
			memcpy(ptr, bits, imageHeight * imageBytesPerLine);
			pboNext->unmap();
		}
		pboNext->release();
	}
#else
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, bits);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGB, GL_UNSIGNED_BYTE, bits);
#endif
}

bool GLSLColorCvt_RGB2RGBA::isColorConversionNeeded()
{
	return false;
}

//////////////
GLSLColorCvt_RGBA2RGBA::GLSLColorCvt_RGBA2RGBA(QObject *parent) : GLSLColorCvt_2RGBA(parent)
{
}

GLSLColorCvt_RGBA2RGBA::~GLSLColorCvt_RGBA2RGBA()
{

}

void GLSLColorCvt_RGBA2RGBA::createSourceGPUResource()
{
	// Source texture
	m_gl->glGenTextures(1, &m_srcTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void GLSLColorCvt_RGBA2RGBA::addShaderFile()
{
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "rgb2rgb");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/rgb2rgb.frag");
#endif
}

void GLSLColorCvt_RGBA2RGBA::addPBO()
{
	for (int i = 0; i < m_targetCount; i++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
		pbo->bind();
		pbo->allocate(imageWidth * imageHeight * 4);
		pbo->release();
		m_pbos.push_back(pbo);
	}
}

void GLSLColorCvt_RGBA2RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	if (index != 0)
		return;
#if 1
	QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
	QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_targetCount];

	pboCurrent->bind();
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageWidth, imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	pboCurrent->release();

	if (imageWidth > 0 && imageHeight > 0)
	{
		pboNext->bind();
		GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::WriteOnly);
		if (ptr)
		{
			memcpy(ptr, bits, imageHeight * imageBytesPerLine);
			pboNext->unmap();
		}
		pboNext->release();
	}
#else
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, bits);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL_RGB, GL_UNSIGNED_BYTE, bits);
#endif
}

bool GLSLColorCvt_RGBA2RGBA::isColorConversionNeeded()
{
	return false;
}

//////////////

GLSLColorCvt_RGB2YUV::GLSLColorCvt_RGB2YUV(QObject *parent) : GPUProgram(parent)
, m_buffer(0)
{
	memset(m_textures, 0, sizeof(m_textures));
}

GLSLColorCvt_RGB2YUV::~GLSLColorCvt_RGB2YUV()
{
	if (m_initialized)
	{
		if (m_textures[0] != 0)
		{
			m_gl->glDeleteTextures(GBUFFER_NUM_TEXTURES, m_textures);
		}
		for (int i = 0; i < m_pbos.size(); i++){
			m_pbos[i]->destroy();
			delete m_pbos[i];
		}
			
		m_pbos.clear();

		delete[] m_buffer;
		m_buffer = 0;
	}
}

void GLSLColorCvt_RGB2YUV::initialize(int panoWidth, int panoHeight)
{
	panoramaWidth = panoWidth;
	panoramaHeight = panoHeight;

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	// frame buffer
	/*m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, panoramaWidth, panoramaHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);*/

	m_gl->glGenTextures(GBUFFER_NUM_TEXTURES, m_textures);
	int widths[3] = { panoramaWidth, panoramaWidth, panoramaWidth };
	int heights[3] = { panoramaHeight, panoramaHeight, panoramaHeight };
	for (unsigned int i = 0; i < GBUFFER_NUM_TEXTURES; i++) {
		m_gl->glBindTexture(GL_TEXTURE_2D, m_textures[i]);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, widths[i], heights[i], 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
		m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, m_textures[i], 0);
	}

	GLenum status = m_gl->glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		//
	}


	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);


	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "rgb2yuv420");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/rgb2yuv420.frag");
#endif
	m_program->link();

	m_program->bind();

	m_program->setUniformValue("texture", 0);


	// PBO for glReadPixels
	for (int i = 0; i < m_targetCount; i++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelPackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::StreamRead);
		pbo->bind();
		pbo->allocate(panoWidth * panoHeight * 1);
		pbo->release();
		m_pbos.push_back(pbo);
	}
	m_targetArrayIndex = 0;

	m_program->release();

	m_buffer = new unsigned char[panoramaWidth * panoramaHeight * 4];

	m_initialized = true;
}

void GLSLColorCvt_RGB2YUV::render(GPUResourceHandle rgbTextureId)
{
	m_program->bind();

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	GLenum DrawBuffers[] = { GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2 };
	//GLenum DrawBuffers[] = { GL_COLOR_ATTACHMENT0 };
	m_functions_2_0->glDrawBuffers(sizeof(DrawBuffers) / sizeof(DrawBuffers[0]), DrawBuffers);

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, panoramaWidth, panoramaHeight);
	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, rgbTextureId);


	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();
}



void GLSLColorCvt_RGB2YUV::getYUVBuffer(unsigned char* yuvBuffer)
{
	m_program->bind();

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	m_targetArrayIndex = (m_targetArrayIndex + 1) % 2;
	for (int i = 0; i < 3; i++)
	{
		QOpenGLBuffer * pboCurrent = m_pbos[i * 2 + m_targetArrayIndex];
		QOpenGLBuffer * pboNext = m_pbos[i * 2 + (m_targetArrayIndex + 1) % 2];

		m_functions_2_0->glReadBuffer(GL_COLOR_ATTACHMENT0 + i);

		pboCurrent->bind();
		m_functions_2_0->glReadPixels(0, 0, panoramaWidth, panoramaHeight, GL_RED, GL_UNSIGNED_BYTE, 0);
		pboCurrent->release();

		pboNext->bind();
		GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::ReadOnly);
		if (ptr)
		{
			memcpy(yuvBuffer + panoramaWidth * panoramaHeight * i, ptr, panoramaWidth * panoramaHeight);
			pboNext->unmap();
		}
		pboNext->release();
	}

	/*m_targetArrayIndex = (m_targetArrayIndex + 1) % m_pboCount;
	QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
	QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_pboCount];

	pboCurrent->bind();
	m_functions_2_0->glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, 0);
	pboCurrent->release();

	pboNext->bind();
	GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::ReadOnly);
	if (ptr)
	{
	memcpy(yuvBuffer, ptr, panoramaWidth * panoramaHeight * 3);
	pboNext->unmap();
	}
	pboNext->release();*/

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_program->release();
}

////

GLSLUniColorCvt::GLSLUniColorCvt() :GPUUniColorCvt()
{

}
GLSLUniColorCvt::~GLSLUniColorCvt()
{

}

bool GLSLUniColorCvt::DynRender(ImageBufferData& img, void* gl)
{
	bool initialized = false;
	if (mFormat == img.mFormat
		&& mConverter != NULL
		&& mConverter->ImageWidth() == img.mImageY.width
		&& mConverter->ImageHeight() == img.mImageY.height)
	{
		initialized = true;
	}

	if (!initialized)
	{	// reset
		Free();

		mFormat = img.mFormat;
		switch (img.mFormat)
		{
		case ImageBufferData::YUV420:
			mConverter = new GLSLColorCvt_YUV2RGBA();
			break;
		case ImageBufferData::YUV422:
			mConverter = new GLSLColorCvt_YUV4222RGBA();
			break;
		case ImageBufferData::YUVJ422P:
			mConverter = new GLSLColorCvt_YUVJ422P2RGBA();
			break;
		case ImageBufferData::RGB888:
			mConverter = new GLSLColorCvt_RGB2RGBA();
			break;
		case ImageBufferData::RGBA8888:
			mConverter = new GLSLColorCvt_RGBA2RGBA();
			break;
		}
		if (mConverter == NULL)
			return false;
		if (gl == NULL)
			gl = QOpenGLContext::currentContext()->functions();
		((GLSLColorCvt_2RGBA *)mConverter)->setGL((QOpenGLFunctions *)gl);
		mConverter->initialize(img.mImageY.width, img.mImageY.height);
	}
	mConverter->render(img);
	return true;
}

// Alpha Texture
GLSLAlphaSource::GLSLAlphaSource(QObject *parent) : GPUAlphaSource()
{
}

GLSLAlphaSource::~GLSLAlphaSource()
{
	if (m_initialized)
	{
		m_gl->glDeleteTextures(m_targetCount, m_fboTextureIds);
		m_gl->glDeleteFramebuffers(m_targetCount, m_fboIds);

		m_gl->glDeleteTextures(1, &m_srcTextureId);

		for (int i = 0; i < m_pbos.size(); i++)
		{
			m_pbos[i]->destroy();
			delete m_pbos[i];
		}
		m_pbos.clear();
	}
}

void GLSLAlphaSource::initialize(int imgWidth, int imgHeight)
{
	imageWidth = 0;
	imageHeight = 0;
	imageBytesPerLine = 0;

	m_gl->glGenTextures(m_targetCount, m_fboTextureIds);
	m_gl->glGenFramebuffers(m_targetCount, m_fboIds);

	for (int i = 0; i < m_targetCount; i++)
	{
		m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureIds[i]);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imgWidth, imgHeight, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);

		// load textures and create framebuffers
		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[i]);
		m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureIds[i], 0);
	}
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
	m_workingGPUResource = 0;

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
	addShaderFile();
	m_program->link();

	m_program->bind();

	m_program->setUniformValue("texture", 0);

	m_targetArrayIndex = 0;

	m_program->release();

	m_initialized = true;
}


void GLSLAlphaSource::render(QImage& img)
{
	m_program->bind();

	int width = img.width();
	int height = img.height();

	if (width == 0 || height == 0) return;

	bool blTextureFormatChanged = false;
	if (imageWidth != width || imageHeight != height)
	{
		imageWidth = width;
		imageHeight = height;
		imageBytesPerLine = width;

		createSourceGPUResource();

		addPBO();

		blTextureFormatChanged = true;
	}
	
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboIds[m_workingGPUResource]);

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, imageWidth, imageHeight);

	const unsigned char * buffer = img.bits();
	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	
	m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
	// host to device buffer uploading.
	setSourceGPUResourceBuffer(buffer, blTextureFormatChanged);
	
	// Don't comment below code block absolutely.
	if (blTextureFormatChanged)
	{	// It's very important to avoid empty-rendering.
		m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
		// host to device buffer uploading.
		setSourceGPUResourceBuffer(buffer, false);
	}

	m_program->setUniformValue("mirrorVert", true);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_workingGPUResource = (m_workingGPUResource + 1) % m_targetCount;

	m_program->release();
}


const int GLSLAlphaSource::getWidth()
{
	return imageWidth;
}

const int GLSLAlphaSource::getHeight()
{
	return imageHeight;
}

void GLSLAlphaSource::createSourceGPUResource()
{
	// Source texture
	m_gl->glGenTextures(1, &m_srcTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_srcTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageBytesPerLine, imageHeight * 3 / 2, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
}

void GLSLAlphaSource::addShaderFile()
{
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "alphaSource");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/alphaSource.frag");
#endif
}

void GLSLAlphaSource::addPBO()
{
	for (int i = 0; i < m_targetCount; i++)
	{
		QOpenGLBuffer* pbo = new QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
		pbo->create();
		pbo->setUsagePattern(QOpenGLBuffer::UsagePattern::DynamicDraw);
		pbo->bind();
		pbo->allocate(imageBytesPerLine * imageHeight);
		pbo->release();
		m_pbos.push_back(pbo);
	}
}

void GLSLAlphaSource::setSourceGPUResourceBuffer(const uchar* bits, bool textureChanged)
{
	buffer_ptr = bits;

	QOpenGLBuffer * pboCurrent = m_pbos[m_targetArrayIndex];
	QOpenGLBuffer * pboNext = m_pbos[(m_targetArrayIndex + 1) % m_targetCount];
	pboCurrent->bind();
	if (textureChanged)
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, getWidth(), getHeight(), 0, GL_RED, GL_UNSIGNED_BYTE, 0);
	else
		m_gl->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, getWidth(), getHeight(), GL_RED, GL_UNSIGNED_BYTE, 0);
	pboCurrent->release();
	if (getWidth() > 0 && getHeight() > 0)
	{
		pboNext->bind();
		GLubyte* ptr = (GLubyte*)pboNext->map(QOpenGLBuffer::WriteOnly);
		if (ptr)
		{
			memcpy(ptr, buffer_ptr, getWidth() * getHeight());
			pboNext->unmap();
		}
		pboNext->release();
	}
}

// Nodal Input
GLSLNodalInput::GLSLNodalInput() :GPUNodalInput()
{
}

GLSLNodalInput::~GLSLNodalInput()
{
}

void GLSLNodalInput::createColorCvt(void* gl, bool liveMode, QString weightFilename)
{
	if (liveMode)
	{
		colorCvt = new GLSLColorCvt_YUV4222RGBA();
	}
	else
	{
		colorCvt = new GLSLColorCvt_YUV2RGBA();
	}
	colorCvt->setGL((QOpenGLFunctions *)gl);

	if (weightFilename != "")
	{
		weightMap = new GLSLAlphaSource();
		weightMap->setGL((QOpenGLFunctions *)gl);
		weightImage.load(weightFilename);
	}
}

void GLSLNodalInput::initialize(int imgWidth, int imgHeight)
{
	colorCvt->initialize(imgWidth, imgHeight);
	if (weightMap)
	{
		weightMap->initialize(weightImage.width(), weightImage.height());
	}
	weightRenderCount = 0;
}


void GLSLNodalInput::render(ImageBufferData& img)
{
	colorCvt->render(img);
	if (weightMap && weightRenderCount < 1)
	{
		weightMap->render(weightImage);
		weightRenderCount++;
	}
}


