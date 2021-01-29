

#include <sstream>
#include <iostream>

#include <iomanip>
#include <string>

#include <QDebug>
#include <QFile>

#include "CaptureImageFile.h"
#include "CaptureProp.h"
#include "define.h"

void CaptureImageFile::init()
{
	m_width = -1;
	m_height = -1;
#ifdef USE_FFMPEG
	av_register_all();
	iformat = av_find_input_format("image2");

	if (!(frame = av_frame_alloc()) || !(frameRGB = av_frame_alloc())) {
		av_log(NULL, AV_LOG_ERROR, "Failed to alloc frame\n");
	}
	data = NULL;
	img_convert_ctx = NULL;
	codec = NULL;
	codec_ctx = NULL;
	int frame_decoded;
	opt = NULL;
#endif
}

QString CaptureImageFile::getFramePath()
{
	if (getCurFrame() == -1)
	{
		return QString("%1/%2.%3").arg(m_fileDir).arg(m_imageFilePrefix).arg(m_ext);
	}
	else
	{
		return QString("%1/%2%3.%4").arg(m_fileDir).arg(m_imageFilePrefix).arg(getCurFrame()).arg(m_ext);
	}
}

bool CaptureImageFile::open(int wIndex)
{
	//std::cout << "Loading : " << sarg.str() << std::endl;

	m_deviceIndex = wIndex;
	QString sarg = getFramePath();
	if (!QFile::exists(sarg))
	{
		qDebug() << QString("[CaptureImageFile] File not exists! -> %1").arg(sarg);
		return false;
	}

#if !defined(USE_FFMPEG)
	if (!(m_ext == "jpg" || m_ext == "png")) {
		return false;
	}

	m_sh = std::shared_ptr<ImageHandler>(new QTHandler);
	
	m_sh->open(sarg.toStdString(), 0, m_width, m_height, m_framerate);
#else
	if (ff_load_image(NULL, &m_width, &m_height, &format, sarg.toLatin1().data(), NULL) >= 0)
	{

		m_sharedImageBuffer->getGlobalAnimSettings()->getCameraInput(wIndex).xres = m_width;
		m_sharedImageBuffer->getGlobalAnimSettings()->getCameraInput(wIndex).yres = m_height;

		int num_bytes = avpicture_get_size(AV_PIX_FMT_RGB24, m_width, m_height);
		data = (uint8_t *)av_malloc(num_bytes);
		avpicture_fill((AVPicture*)frameRGB, data, AV_PIX_FMT_RGB24, m_width, m_height);
		img_convert_ctx = sws_getCachedContext(NULL, m_width, m_height, format, m_width, m_height,
			AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
	}
	else
	{
		return false;
	}
#endif

	return true;
}


void CaptureImageFile::close()
{
#if !defined(USE_FFMPEG)
	m_sh->close();
#else
	av_free(data);
	if (img_convert_ctx)	sws_freeContext(img_convert_ctx);
	if (frame)				av_frame_free(&frame);
	if (frameRGB)			av_frame_free(&frameRGB);
	if (opt)				av_dict_free(&opt);
#endif
}


bool CaptureImageFile::grabFrame(ImageBufferData& frame)
{
	m_incomingType = D360::Capture::None;

	QString sarg = getFramePath();
	if (!QFile::exists(sarg)) {
		qDebug() << QString("[CaptureImageFile] File not exists! -> %1").arg(sarg);
		return false;
	}

	int ret = ff_load_image(data, NULL, NULL, NULL, sarg.toLatin1().data(), NULL);
	this->m_incomingType = D360::Capture::Video;

#ifdef _DEBUG
	qDebug() << "Current Frame: " << sarg;
#endif

	return true;
}

bool CaptureImageFile::retrieveFrame(int channel, ImageBufferData& frame)
{
	if (m_incomingType == IncomingFrameType::Video)
	{
#if !defined(USE_FFMPEG)
		if (!(m_ext == "jpg" || m_ext == "png"))	return false;

		m_sh->read(getCurFrame(), imgY, sarg);
		imgU = AlignedImage();
		imgV = AlignedImage();
#else
		frame.mFormat = ImageBufferData::RGB888;
		frame.mImageY.buffer = data;
		frame.mImageY.width = m_width;
		frame.mImageY.height = m_height;
		frame.mImageY.stride = m_width * 3;
		frame.mImageU = AlignedImage();
		frame.mImageV = AlignedImage();
#endif

		m_incomingType = IncomingFrameType::None;
		return true;
	}
	return false;
}


double CaptureImageFile::getProperty(int property_id)
{
	switch (property_id)
	{
	case CV_CAP_PROP_FRAME_WIDTH:
		return (double)m_width;
		break;
	case CV_CAP_PROP_FRAME_HEIGHT:
		return (double)m_height;
		break;
	case CV_CAP_PROP_FPS:
		return (double)m_framerate;
	default:
		return 0.0;
	}
}

bool CaptureImageFile::setProperty(int property_id, double value)
{
	switch (property_id)
	{
	case CV_CAP_PROP_FRAME_WIDTH:
		m_width = (int)value;
		break;
	case CV_CAP_PROP_FRAME_HEIGHT:
		m_height = (int)value;
		break;
	case CV_CAP_PROP_FPS:
		m_framerate = (float)value;
		break;
	}

	return true;
}
#ifdef USE_FFMPEG
int CaptureImageFile::ff_load_image(uint8_t *data,
	int *w, int *h, enum AVPixelFormat *pix_fmt,
	const char *filename, void *log_ctx)
{
	int ret = 0;

	AVFormatContext *format_ctx = NULL;
	if ((ret = avformat_open_input(&format_ctx, filename, iformat, NULL)) < 0) {
		av_log(log_ctx, AV_LOG_ERROR,
			"Failed to open input file '%s'\n", filename);
		return ret;
	}

	if ((ret = avformat_find_stream_info(format_ctx, NULL)) < 0) {
		av_log(log_ctx, AV_LOG_ERROR, "Find stream info failed\n");
		return ret;
	}

	av_init_packet(&pkt);

	codec_ctx = format_ctx->streams[0]->codec;
	//AVCodecID codec_id = format_ctx->streams[0]->codecpar.codec_id;
	codec = avcodec_find_decoder(codec_ctx->codec_id);
	//codec = avcodec_find_decoder(AV_CODEC_ID_JPEG2000);
	if (!codec) {
		av_log(log_ctx, AV_LOG_ERROR, "Failed to find codec\n");
		ret = AVERROR(EINVAL);
		goto end;
	}

	av_dict_set(&opt, "thread_type", "slice", 0);
	if ((ret = avcodec_open2(codec_ctx, codec, &opt)) < 0) {
		av_log(log_ctx, AV_LOG_ERROR, "Failed to open codec\n");
		goto end;
	}

	ret = av_read_frame(format_ctx, &pkt);
	if (ret < 0) {
		av_log(log_ctx, AV_LOG_ERROR, "Failed to read frame from file\n");
		goto end;
	}

	ret = avcodec_decode_video2(codec_ctx, frame, &frame_decoded, &pkt);
	//ret = avcodec_receive_frame(codec_ctx, frame);
	if (ret < 0 || !frame_decoded) {
		av_log(log_ctx, AV_LOG_ERROR, "Failed to decode image from file\n");
		if (ret >= 0)
			ret = -1;
		goto end;
	}

	if (w)			*w = frame->width;
	if (h)			*h = frame->height;
	if (pix_fmt)	*pix_fmt = (AVPixelFormat)frame->format;

	ret = 0;

	if (data) {
		//av_image_copy(data, linesize, (const uint8_t **)frame->data, frame->linesize, (AVPixelFormat)frame->format, frame->width, frame->height);
		sws_scale(img_convert_ctx, ((AVPicture*)frame)->data, ((AVPicture*)frame)->linesize, 0, frame->height, ((AVPicture *)frameRGB)->data, ((AVPicture *)frameRGB)->linesize);
	}

end:
	av_free_packet(&pkt);
	avcodec_close(codec_ctx);
	avformat_close_input(&format_ctx);
	if (ret < 0)
		av_log(log_ctx, AV_LOG_ERROR, "Error loading image file '%s'\n", filename);
	return ret;
}
#endif
