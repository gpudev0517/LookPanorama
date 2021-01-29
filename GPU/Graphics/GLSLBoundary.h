#ifndef GLSLBOUNDARY_H
#define GLSLBOUNDARY_H

#include "GPUBoundary.h"

/// <summary>
/// The opengl shader that calculates the boundary area for each view.
/// </summary>
class GLSLBoundary : public GPUBoundary
{
	Q_OBJECT
public:
	explicit GLSLBoundary(QObject *parent = 0);
	virtual ~GLSLBoundary();

	virtual void initialize(int panoWidth, int panoHeight, int viewCount);
	virtual void render(GPUResourceHandle *fbos);
};

#endif // GLSLBOUNDARY_H