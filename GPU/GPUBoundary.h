#ifndef GPUBOUNDARY_H
#define GPUBOUNDARY_H

#include "GPUProgram.h"

/// <summary>
/// The opengl shader that calculates the boundary area for each view.
/// </summary>
class GPUBoundary : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUBoundary(QObject *parent = 0);
	virtual ~GPUBoundary();

	virtual void initialize(int panoWidth, int panoHeight, int viewCount) = 0;
	virtual void render(GPUResourceHandle *fbos) = 0;

	virtual const int getWidth();
	virtual const int getHeight();

protected:
	int panoramaWidth;
	int panoramaHeight;

	int m_viewCount;
	int m_viewIdx;
};

#endif // GPUBOUNDARY_H