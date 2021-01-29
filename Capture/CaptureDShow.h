#pragma once

#include <memory>

#include <QImage>

#include "Capture.h"
#include "CaptureAudio.h"
#include "Structures.h"
#include "SharedImageBuffer.h"
#include "CaptureThread.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavutil/mem.h>
#include <libswscale/swscale.h>
}

#include <iostream>

struct AVFrameBuffer
{
	void alloc(std::size_t cap);
	void clear();

	AVFrame* getNext();
	AVFrame* getFirst();

private:
	std::vector<AVFrame*> buffer;
	std::size_t first = 0;
	std::size_t last = 0;
};

/**********************************************************************************/

class CaptureDShow : public D360::Capture, public AudioInput
{
public:
	CaptureDShow(SharedImageBuffer *sharedImageBuffer):
	m_sharedImageBuffer(sharedImageBuffer)
	{
		m_captureDomain = D360::Capture::CAPTURE_DSHOW;
		init();
		m_Name = "CaptureDShow";
	}
	virtual ~CaptureDShow()
	{
		close();
	}

	virtual void		reset(ImageBufferData& frame);
	int					getTotalFrames() { return m_nTotalFrame; }
	virtual void		seekFrames(int nFrames);
	bool				open(int index, QString name, int width, int height, float fps, Capture::CaptureDomain captureType);
	virtual void		close();
	virtual double		getProperty(int);
	virtual bool		grabFrame(ImageBufferData& frame);
	virtual bool		retrieveFrame(int channel, ImageBufferData& frame);
	virtual void		seekFirst();
	virtual void*		retrieveAudioFrame();
	void				setAudio(CameraInput::InputAudioChannelType audioType);
	virtual ImageBufferData convertToRGB888(ImageBufferData& image);

private:
	void init();
	int decode_packet(AVPacket pkt, int *got_frame);

	SharedImageBuffer *m_sharedImageBuffer;
	int m_nCameraIndex;

	AVCodecContext* m_pVideoCodecCtx;
	AVFormatContext* m_pFormatCtx;
	int m_nVideoStream;
	int m_nAudioStream;
	int m_nTotalFrame;
	SwsContext * img_convert_ctx;

	//AVFrame* m_pVideoFrame; // decoded frame (grab)
	AVFrameBuffer m_videoFrameBuffer;
	AVFrame* m_pFrameRGB; // rgb frame for unsupported gpu format
	uint8_t* m_pFrameBuffer;

	QString m_cDeviceName;
	QString m_aDeviceName;

	int m_nWidth;
	int m_nHeight;

	float m_fps;
	bool m_isAudioDisabled;
	bool m_isLeft;
	bool m_isRight;
	QString m_Name;
};
