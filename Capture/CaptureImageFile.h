#pragma once

#include <memory>

#include <QImage>

#include "Capture.h"

#include "QTHandler.h"

#include <iostream>

#define USE_FFMPEG

#ifdef USE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavutil/mem.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavformat/avformat.h>
}
#endif

#include "SharedImageBuffer.h"

/**********************************************************************************/

class CaptureImageFile : public D360::Capture
{
public:
	CaptureImageFile(SharedImageBuffer *sharedImageBuffer) :
		m_sharedImageBuffer(sharedImageBuffer)
	{
		m_captureDomain = D360::Capture::CAPTURE_FILE;
		init();
	}
	virtual ~CaptureImageFile()
	{
		close();
	}

	virtual bool		open(int index);
	virtual void		close();
	virtual double		getProperty(int);
	virtual bool		setProperty(int, double);
	virtual bool		grabFrame(ImageBufferData& frame);
	virtual bool		retrieveFrame(int channel, ImageBufferData& frame);

	void setImageFileDir(QString fileDir)
	{
		m_fileDir = fileDir;
	}
	void setImageFilePrefix(QString prefix)
	{
		m_imageFilePrefix = prefix;
	}

	void setImageFileExt(QString ext)
	{
		m_ext = ext;
	}

	int msToWait()
	{
#ifdef USE_FFMPEG
		return 0;
#else
		return m_sh->getMsToWait();
#endif
	}
private:
	void init();
	void errMsg(const char* msg, int errNum);

	int  getBpp();

#ifdef USE_FFMPEG
	int ff_load_image(uint8_t *data,
		int *w, int *h, enum AVPixelFormat *pix_fmt,
		const char *filename, void *log_ctx);

	SharedImageBuffer *m_sharedImageBuffer;

	AVInputFormat *iformat;
	AVCodec *codec;
	AVCodecContext *codec_ctx;
	AVFrame *frame;
	AVFrame *frameRGB;
	int frame_decoded;
	AVPacket pkt;
	AVDictionary *opt;
	SwsContext * img_convert_ctx;

	uint8_t *data;
	enum AVPixelFormat format;
#endif

	int m_deviceIndex;
	QString m_fileDir;
	QString m_imageFilePrefix;
	QString m_ext;

	int m_width;
	int m_height;

	float m_framerate;
	//NorPixSeqHandler m_sh;
	std::shared_ptr< ImageHandler > m_sh;

	QString getFramePath();
};
