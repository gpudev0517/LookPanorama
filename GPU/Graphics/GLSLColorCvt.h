#ifndef GLSLCOLORCVT_H
#define GLSLCOLORCVT_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLFunctions_2_0>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>

#include "GPUProgram.h"
#include "GPUColorCvt.h"
#include "D360Parser.h"
#include "SharedImageBuffer.h"
#include "define.h"


class GLSLColorCvt_2RGBA : public GPUColorCvt_2RGBA
{
	Q_OBJECT
public:
	GLSLColorCvt_2RGBA(QObject *parent = 0);
	virtual ~GLSLColorCvt_2RGBA();

	void initialize(int imgWidth, int imgHeight);
	virtual void render(ImageBufferData& img);

	GPUResourceHandle getTargetGPUResource() { return m_fboTextureIds[getReadyGPUResourceIndex()]; }

protected:

	virtual void addShaderFile() = 0;
	virtual void addPBO() = 0;

protected:

	GLuint m_srcTextureId;
	

	GLuint m_fboIds[m_targetCount];
	GLuint m_fboTextureIds[m_targetCount];
	
	// PBO
	std::vector<QOpenGLBuffer*> m_pbos;
	
	GLuint bytesPerLineUnif;
	GLuint imageWidthUnif;
	GLuint imageHeightUnif;
};

// convert yuv420 buffer to rgb buffer. This is common case for importing video files.
class GLSLColorCvt_YUV2RGBA : public GLSLColorCvt_2RGBA
{
    Q_OBJECT
public:
	explicit GLSLColorCvt_YUV2RGBA(QObject *parent = 0);
	virtual ~GLSLColorCvt_YUV2RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void addShaderFile();
	virtual void addPBO();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();

private:
	const uchar* buffer_ptrs[3];
};

// convert yuv422 buffer to rgb buffer. This is for standard cameras.
class GLSLColorCvt_YUV4222RGBA : public GLSLColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit GLSLColorCvt_YUV4222RGBA(QObject *parent = 0);
	virtual ~GLSLColorCvt_YUV4222RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void addShaderFile();
	virtual void addPBO();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();
};

// convert yuvj422p buffer to rgb buffer. This is for logitech camera.
class GLSLColorCvt_YUVJ422P2RGBA : public GLSLColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit GLSLColorCvt_YUVJ422P2RGBA(QObject *parent = 0);
	virtual ~GLSLColorCvt_YUVJ422P2RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void addShaderFile();
	virtual void addPBO();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();

private:
	const uchar* buffer_ptrs[3];
};

class GLSLColorCvt_BGR02RGBA : public GLSLColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit GLSLColorCvt_BGR02RGBA(QObject *parent = 0);
	virtual ~GLSLColorCvt_BGR02RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void addShaderFile();
	virtual void addPBO();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();
};

class GLSLColorCvt_BAYERRG82RGBA : public GLSLColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit GLSLColorCvt_BAYERRG82RGBA(QObject *parent = 0);
	virtual ~GLSLColorCvt_BAYERRG82RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void addShaderFile();
	virtual void addPBO();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();
};

class GLSLColorCvt_RGB2RGBA : public GLSLColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit GLSLColorCvt_RGB2RGBA(QObject *parent = 0);
	virtual ~GLSLColorCvt_RGB2RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void addShaderFile();
	virtual void addPBO();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();
};

class GLSLColorCvt_RGBA2RGBA : public GLSLColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit GLSLColorCvt_RGBA2RGBA(QObject *parent = 0);
	virtual ~GLSLColorCvt_RGBA2RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void addShaderFile();
	virtual void addPBO();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();
};

// convert rgb buffer to yuv420 so that we can directly broadcast it to the wowza server
class GLSLColorCvt_RGB2YUV : public GPUProgram
{
	Q_OBJECT
public:
	enum GBUFFER_TEXTURE_TYPE {
		GBUFFER_TEXTURE_TYPE_Y,
		GBUFFER_TEXTURE_TYPE_U,
		GBUFFER_TEXTURE_TYPE_V,
		GBUFFER_NUM_TEXTURES
	};

	explicit GLSLColorCvt_RGB2YUV(QObject *parent = 0);
	virtual ~GLSLColorCvt_RGB2YUV();

	void initialize(int panoWidth, int panoHeight);
	void render(GPUResourceHandle rgbTextureId);

	int getFBOId() { return m_fboId; }
	void getYUVBuffer(unsigned char* yuvBuffer);

private:

	int panoramaWidth;
	int panoramaHeight;

	// color texture
	GLuint m_textures[GBUFFER_NUM_TEXTURES];

	unsigned char * m_buffer;

	const int m_targetCount = 6;
	std::vector<QOpenGLBuffer*> m_pbos;
	int m_targetArrayIndex;
};

class GLSLUniColorCvt : public GPUUniColorCvt
{
public:
	GLSLUniColorCvt();
	virtual ~GLSLUniColorCvt();
	// render
	virtual bool DynRender(ImageBufferData& img, void *gl=NULL);
	
};

// Alpha Texture
class GLSLAlphaSource : public GPUAlphaSource
{
	Q_OBJECT
public:
	GLSLAlphaSource(QObject *parent = 0);
	virtual ~GLSLAlphaSource();

	virtual void initialize(int imgWidth, int imgHeight);
	virtual void render(QImage& img);

	const int getWidth();
	const int getHeight();

	GPUResourceHandle getTargetGPUResource() { return m_fboTextureIds[getReadyGPUResourceIndex()]; }
protected:
	virtual void createSourceGPUResource();
	virtual void addShaderFile();
	virtual void addPBO();
	virtual void setSourceGPUResourceBuffer(const uchar* bits, bool textureChanged);


public:
	int ImageWidth() const { return imageWidth; }
	int ImageHeight() const { return imageHeight; }
protected:

	GLuint m_srcTextureId;

	
	GLuint m_fboIds[m_targetCount];
	GLuint m_fboTextureIds[m_targetCount];

	// PBO
	std::vector<QOpenGLBuffer*> m_pbos;

	GLuint bytesPerLineUnif;
	GLuint imageWidthUnif;
	GLuint imageHeightUnif;
};

// Nodal Input

class GLSLNodalInput : public GPUNodalInput
{
public:
	GLSLNodalInput();
	virtual ~GLSLNodalInput();
	void createColorCvt(void* gl, bool liveMode, QString weightFilename);
	void initialize(int imgWidth, int imgHeight);
	void render(ImageBufferData& img);
	int getWeightTexture();
};

#endif // GLSLCOLORCVT_H