#ifndef GPUSEAM_H
#define GPUSEAM_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLFunctions_2_0>
#include <QOpenGLTexture>

#include "GLSLBoundary.h"
#ifdef USE_CUDA
#include "CUDABoundary.h"
#endif

class GPUSeamMask;
class GPUSeamRegion;

/// <summary>
/// The opengl shader that calculates mask, boundary, and seam area.
/// </summary>
class GPUSeam : public QObject
{
	Q_OBJECT
public:
	explicit GPUSeam(QObject *parent = 0);
	virtual ~GPUSeam();

	void setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0);
	void initialize(int panoWidth, int panoHeight, int viewCount);
	void render(GPUResourceHandle *weighttextures, bool weightMapChanged);

	int getSeamTexture();
	GPUResourceHandle getBoundaryTexture();

	void setSeamIndex(int index);
	int getViewCount() { return m_viewCount; }

private:
	int m_panoramaWidth;
	int m_panoramaHeight;
	
	// gl functions
	QOpenGLFunctions* m_gl;
	QOpenGLFunctions_2_0* m_functions_2_0;

	GPUSeamRegion *m_seamRegion;
	GPUSeamMask *m_seamMask;

	bool m_initialized;
	int m_viewCount;

	bool isSeamIdxChanged;
	int m_showSeamIdx;
};

class GPUSeamRegion : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUSeamRegion(QObject *parent = 0);
	virtual ~GPUSeamRegion();

	virtual void initialize(int panoWidth, int panoHeight, int viewCount) = 0;
	virtual void render(GPUResourceHandle *weighttextures);
	virtual void getCameraRegionImage(int seamIdx) = 0;

	const int getWidth();
	const int getHeight();

	GPUResourceHandle getBoundaryTexture() { return m_boundary->getTargetGPUResource(); }


protected:

	int m_panoramaWidth;
	int m_panoramaHeight;

	int m_viewCount;

	GPUBoundary* m_boundary;
};


class GLSLSeamRegion : public GPUSeamRegion
{
	Q_OBJECT
public:
	explicit GLSLSeamRegion(QObject *parent = 0);
	virtual ~GLSLSeamRegion();

	void initialize(int panoWidth, int panoHeight, int viewCount);
	void getCameraRegionImage(int seamIdx);
};

#ifdef USE_CUDA
class CUDASeamRegion : public GPUSeamRegion
{
	Q_OBJECT
public:
	explicit CUDASeamRegion(QObject *parent = 0);
	virtual ~CUDASeamRegion();

	void initialize(int panoWidth, int panoHeight, int viewCount);
	void getCameraRegionImage(int seamIdx);
};

#endif

class GPUSeamMask : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUSeamMask(QObject *parent = 0);
	virtual ~GPUSeamMask();

	virtual void initialize(int panoWidth, int panoHeight) = 0;
	virtual void getSeamMaskImage(GPUResourceHandle regionTexture) = 0;

	const int getWidth();
	const int getHeight();

protected:

	int m_panoramaWidth;
	int m_panoramaHeight;
};

class GLSLSeamMask : public GPUSeamMask
{
	Q_OBJECT
public:
	explicit GLSLSeamMask(QObject *parent = 0);
	virtual ~GLSLSeamMask();

	virtual void initialize(int panoWidth, int panoHeight);
	virtual void getSeamMaskImage(GPUResourceHandle regionTexture);
};

#ifdef USE_CUDA
class CUDASeamMask : public GPUSeamMask
{
	Q_OBJECT
public:
	explicit CUDASeamMask(QObject *parent = 0);
	virtual ~CUDASeamMask();

	virtual void initialize(int panoWidth, int panoHeight);
	virtual void getSeamMaskImage(GPUResourceHandle regionTexture);
};
#endif
#endif // GPUSEAM_H
