#ifndef CUDABOUNDARY_H
#define CUDABOUNDARY_H

#ifdef USE_CUDA
#include "GPUProgram.h"
#include "GPUBoundary.h"

/// <summary>
/// The opengl shader that calculates the boundary area for each view.
/// </summary>
class CUDABoundary : public GPUBoundary
{
	Q_OBJECT
public:
	explicit CUDABoundary(QObject *parent = 0);
	virtual ~CUDABoundary();

	void initialize(int panoWidth, int panoHeight, int viewCount);
	void render(GPUResourceHandle fbos[]);

private:

};
#endif //USE_CUDA

#endif // CUDABOUNDARY_H