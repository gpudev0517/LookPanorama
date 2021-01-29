#include "GPUNodalBlending.h"
#include "define.h"
#include "common.h"

GPUNodalBlending::GPUNodalBlending(QObject *parent) : GPUProgram(parent)
{
}

GPUNodalBlending::~GPUNodalBlending()
{
	if (m_initialized)
	{
		m_initialized = false;
	}
}
