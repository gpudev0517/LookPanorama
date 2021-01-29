#ifndef GPUPROGRAM_H
#define GPUPROGRAM_H

#include <QtGui>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLFunctions_2_0>
#include <QtGui/QOpenGLFunctions_4_3_Compatibility>
#include <QOpenGLTexture>

#include "common.h"

#ifdef USE_CUDA
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

class GPUProgram : public QObject
{
	Q_OBJECT

public:
	GPUProgram(QObject *parent = 0);
	virtual ~GPUProgram();

	virtual void initialize();
	void setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0 = NULL, QOpenGLFunctions_4_3_Compatibility* functions_4_3 = NULL);

 	virtual GPUResourceHandle getTargetGPUResource();
	virtual GPUResourceHandle getTargetBuffer();

	virtual bool isInitialized() { return m_initialized; }
	static void ADD_SHADER_FROM_CODE(QOpenGLShaderProgram* program, QString type, QString res);

protected:

	//Both OpenGL and CUDA
	bool m_initialized;

#ifdef USE_CUDA
	//Only CUDA
	cudaTextureObject_t m_cudaTargetTexture;
	cudaSurfaceObject_t m_cudaTargetSurface;
	cudaArray *m_cudaTargetArray;
#endif

	//Only OpenGL
	// GLSL program
	GLuint m_fboTextureId;
	QOpenGLShaderProgram *m_program;

	GLuint m_vertexAttr;
	GLuint m_texCoordAttr;

	// framebuffer index and color texture
	GLuint m_fboId;

	// gl functions
	QOpenGLFunctions* m_gl;
	QOpenGLFunctions_2_0* m_functions_2_0;
	QOpenGLFunctions_4_3_Compatibility* m_functions_4_3;
	
};

#endif // GPUPROGRAM_H