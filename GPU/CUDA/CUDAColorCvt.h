#ifndef CUDACOLORCVT_H
#define CUDACOLORCVT_H

#ifdef USE_CUDA
#include <QtGui>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "GPUColorCvt.h"
#include "D360Parser.h"
#include "SharedImageBuffer.h"
#include "define.h"


class CUDAColorCvt_2RGBA : public GPUColorCvt_2RGBA
{
	Q_OBJECT
public:
	CUDAColorCvt_2RGBA(QObject *parent = 0);
	virtual ~CUDAColorCvt_2RGBA();

	void initialize(int imgWidth, int imgHeight);
	virtual void render(ImageBufferData& img);
	virtual void runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline) = 0;

	GPUResourceHandle getTargetGPUResource() { return m_cudaTargetTextures[getReadyGPUResourceIndex()]; }

protected:
	cudaSurfaceObject_t m_cudaSrcSurfaces[m_targetCount];
	cudaArray *m_cudaSrcArrays[m_targetCount];
	cudaTextureObject_t m_cudaSrcTextures[m_targetCount];

	cudaSurfaceObject_t m_cudaTargetSurfaces[m_targetCount];
	cudaArray *m_cudaTargetArrays[m_targetCount];
	cudaTextureObject_t m_cudaTargetTextures[m_targetCount];
};

// convert yuv420 buffer to rgb buffer. This is common case for importing video files.
class CUDAColorCvt_YUV2RGBA : public CUDAColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit CUDAColorCvt_YUV2RGBA(QObject *parent = 0);
	virtual ~CUDAColorCvt_YUV2RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();

	virtual void runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);

private:
	const uchar* buffer_ptrs[3];
};

// convert yuv422 buffer to rgb buffer. This is usual 
class CUDAColorCvt_YUV4222RGBA : public CUDAColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit CUDAColorCvt_YUV4222RGBA(QObject *parent = 0);
	virtual ~CUDAColorCvt_YUV4222RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();

	virtual void runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);
};

class CUDAColorCvt_YUVJ422P2RGBA : public CUDAColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit CUDAColorCvt_YUVJ422P2RGBA(QObject *parent = 0);
	virtual ~CUDAColorCvt_YUVJ422P2RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();

	virtual void runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);

private:
	const uchar* buffer_ptrs[3];
};

class CUDAColorCvt_BGR02RGBA : public CUDAColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit CUDAColorCvt_BGR02RGBA(QObject *parent = 0);
	virtual ~CUDAColorCvt_BGR02RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();

	virtual void runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);
};

class CUDAColorCvt_RGB2RGBA : public CUDAColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit CUDAColorCvt_RGB2RGBA(QObject *parent = 0);
	virtual ~CUDAColorCvt_RGB2RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();

	virtual void runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);
};

class CUDAColorCvt_RGBA2RGBA : public CUDAColorCvt_2RGBA
{
	Q_OBJECT
public:
	explicit CUDAColorCvt_RGBA2RGBA(QObject *parent = 0);
	virtual ~CUDAColorCvt_RGBA2RGBA();

protected:
	virtual void createSourceGPUResource();
	virtual void setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged);
	virtual bool isColorConversionNeeded();

	virtual void runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);
};
// 
// // convert rgb buffer to yuv420 so that we can directly broadcast it to the wowza server
// class GLSLColorCvt_RGB2YUV : public QObject
// {
// 	Q_OBJECT
// public:
// 	enum GBUFFER_TEXTURE_TYPE {
// 		GBUFFER_TEXTURE_TYPE_Y,
// 		GBUFFER_TEXTURE_TYPE_U,
// 		GBUFFER_TEXTURE_TYPE_V,
// 		GBUFFER_NUM_TEXTURES
// 	};
// 
// 	explicit GLSLColorCvt_RGB2YUV(QObject *parent = 0);
// 	virtual ~GLSLColorCvt_RGB2YUV();
// 
// 	void setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0);
// 	void initialize(int panoWidth, int panoHeight);
// 	void render(GPUResourceHandle rgbTextureId);
// 
// 	int getFBOId() { return m_fboId; }
// 	void getYUVBuffer(unsigned char* yuvBuffer);
// 
// private:
// 
// 	int panoramaWidth;
// 	int panoramaHeight;
// 
// 	// framebuffer index and color texture
// 	GLuint m_fboId;
// 	//GLuint m_fboTextureId;
// 	GLuint m_textures[GBUFFER_NUM_TEXTURES];
// 
// 	// undistort GLSL program
// 	QOpenGLShaderProgram *m_program;
// 
// 	GLuint m_vertexAttr;
// 	GLuint m_texCoordAttr;
// 
// 	// gl functions
// 	QOpenGLFunctions* m_gl;
// 	QOpenGLFunctions_2_0* m_functions_2_0;
// 
// 	unsigned char * m_buffer;
// 
// 	const int m_targetCount = 6;
// 	std::vector<QOpenGLBuffer*> m_pbos;
// 	int m_targetArrayIndex;
// 
// 	bool m_initialized;
// };

class CUDAUniColorCvt : public GPUUniColorCvt
{
public:
	CUDAUniColorCvt();
	virtual ~CUDAUniColorCvt();
	// render
	virtual bool DynRender(ImageBufferData& img, void *gl = NULL);

};

// Alpha Texture
class CUDAAlphaSource : public GPUAlphaSource
{
	Q_OBJECT
public:
	CUDAAlphaSource(QObject *parent = 0);
	virtual ~CUDAAlphaSource();

	virtual void initialize(int imgWidth, int imgHeight);
	virtual void render(QImage& img);

	const int getWidth();
	const int getHeight();

	GPUResourceHandle getTargetGPUResource() { return m_cudaTargetTextures[getReadyGPUResourceIndex()]; }
protected:
	virtual void createSourceGPUResource();
	virtual void setSourceGPUResourceBuffer(const uchar* bits, bool textureChanged);


public:
	int ImageWidth() const { return imageWidth; }
	int ImageHeight() const { return imageHeight; }
protected:
	cudaSurfaceObject_t m_cudaSrcSurfaces[m_targetCount];
	cudaArray *m_cudaSrcArrays[m_targetCount];
	cudaTextureObject_t m_cudaSrcTextures[m_targetCount];

	cudaSurfaceObject_t m_cudaTargetSurfaces[m_targetCount];
	cudaArray *m_cudaTargetArrays[m_targetCount];
	cudaTextureObject_t m_cudaTargetTextures[m_targetCount];

};

// Nodal Input

class CUDANodalInput : public GPUNodalInput
{
public:
	CUDANodalInput();
	virtual ~CUDANodalInput();
	void createColorCvt(void* gl, bool liveMode, QString weightFilename);
	void initialize(int imgWidth, int imgHeight);
	void render(ImageBufferData& img);
};

#endif //USE_CUDA

#endif // CUDACOLORCVT_H