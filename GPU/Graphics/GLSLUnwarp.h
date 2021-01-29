#ifndef GLSLUNWARP_H
#define GLSLUNWARP_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>

#include "GPUProgram.h"

#include "D360Parser.h"
#include "common.h"
#include "GPUUnwarp.h"

// undistorts fisheye-lens camera image to panorama image coordinate and render it to FrameBuffer
class GLSLUnwarp : public GPUUnwarp
{
    Q_OBJECT
public:
    explicit GLSLUnwarp(QObject *parent = 0, bool isYUV = true);
	virtual ~GLSLUnwarp();

	virtual void initialize(int id, int xres, int yres, int panoWidth, int panoHeight);
	virtual void render(GPUResourceHandle rgbTextureId, RenderMode renderMode);
	virtual void updateCameraParams();
	
private:
	CameraData cam;
	GLuint ubo;
};

#endif // GLSLUNWARP_H
