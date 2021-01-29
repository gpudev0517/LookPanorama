#ifndef CAPTUREPTGREY_H
#define CAPTUREPTGREY_H

#pragma once

#include <memory>

#include <QImage>

#include "Capture.h"
#include "CaptureAudio.h"
#include "Structures.h"
#include "SharedImageBuffer.h"
#include "CaptureThread.h"

#include <iostream>
/**********************************************************************************/

static SystemPtr g_system = NULL;
static int g_nReferred = 0;
class CapturePtGrey : public D360::Capture
{
public:
    CapturePtGrey(SharedImageBuffer *sharedImageBuffer):
    m_sharedImageBuffer(sharedImageBuffer)
    {
        m_captureDomain = D360::Capture::CAPTURE_PTGREY;
        init();
    }
    virtual ~CapturePtGrey()
    {
        close();
    }

    virtual void		reset(ImageBufferData& frame);
    bool				open(int index, QString name, int width, int height, Capture::CaptureDomain captureType);
    virtual void		close();
    virtual double		getProperty(int);
    virtual bool		grabFrame(ImageBufferData& frame);
    virtual bool		retrieveFrame(int channel, ImageBufferData& frame);

private:
    void init();

    ImagePtr m_pImage; // raw image from the camera
    CameraPtr m_pCamera;
    SharedImageBuffer *m_sharedImageBuffer;
    int m_nCameraIndex;
};

#endif // CAPTUREPTGREY_H
