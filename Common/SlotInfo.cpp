#include "SlotInfo.h"
#include "define.h"
#include "CaptureDShow.h"
#include "QmlMainWindow.h"

extern QmlMainWindow* g_mainWindow;

static int open_codec_context(int *stream_idx,
	AVFormatContext *fmt_ctx, enum AVMediaType type)
{
	int ret, stream_index;
	AVStream *st;
	AVCodecContext *dec_ctx = NULL;
	AVCodec *dec = NULL;
	AVDictionary *opts = NULL;

	ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
	if (ret < 0) {
		fprintf(stderr, "Could not find %s stream from device\n",
			av_get_media_type_string(type));
		return ret;
	}
	else {
		stream_index = ret;
		st = fmt_ctx->streams[stream_index];

		/* find decoder for the stream */
		dec_ctx = st->codec;
		dec = avcodec_find_decoder(dec_ctx->codec_id);
		if (!dec) {
			fprintf(stderr, "Failed to find %s codec\n",
				av_get_media_type_string(type));
			return AVERROR(EINVAL);
		}

		/* Init the decoders, with or without reference counting */
		//av_dict_set(&opts, "refcounted_frames", refcount ? "1" : "0", 0);
		if ((ret = avcodec_open2(dec_ctx, dec, &opts)) < 0) {
			fprintf(stderr, "Failed to open %s codec\n",
				av_get_media_type_string(type));
			return ret;
		}
		*stream_idx = stream_index;
	}

	return 0;
}

SlotInfo::SlotInfo(QObject *parent, int type) : QObject(parent)
, m_type((D360::Capture::CaptureDomain)type)
{
	m_Name = "SlotInfo";
	m_width = 0;
	m_height = 0;
	m_rate = 0;

	av_register_all();
	avdevice_register_all();
	avcodec_register_all();
	m_pFormatCtx = avformat_alloc_context();
}

int SlotInfo::open(QString name)
{
	QString cDeviceName;
	QString fullDName;
	AVInputFormat *fmt = NULL;
	int dupIndex = 0;
	if (m_type == D360::Capture::CAPTURE_DSHOW)
	{
		QString deviceName = g_mainWindow->getDeviceNameByDevicePath(name, dupIndex);
		fullDName = cDeviceName.prepend("video=") + ":" + deviceName;
		fmt = av_find_input_format("dshow");
	}
	else
		fullDName = name;

	PANO_LOG("Path of DirectShow slot: " + fullDName);

	int ret, i;
	int m_nVideoStream = -1;

	AVDictionary        *tOptions = NULL;
	//av_dict_set(&tOptions, "framerate", "30", 0);
	av_dict_set(&tOptions, "video_device_number", QString::number(dupIndex).toLocal8Bit().data(), 0);

	int errCode = avformat_open_input(&m_pFormatCtx, fullDName.toUtf8().data(), fmt, &tOptions);
	if (errCode < 0)
	{
		PANO_ERROR(QString("Could not open device '%1', Error Code = %2").arg(fullDName).arg(errCode));
		close();
		return -1;
	}
	if ((ret = avformat_find_stream_info(m_pFormatCtx, 0)) < 0)
	{
		PANO_ERROR("Failed to retrieve input stream information");
		close();
		return -2;
	}

	if (open_codec_context(&m_nVideoStream, m_pFormatCtx, AVMEDIA_TYPE_VIDEO) < 0)
	{
		close();
		return -3;
	}

	AVCodecContext* m_pVideoCodecCtx = m_pFormatCtx->streams[m_nVideoStream]->codec;

	m_width = m_pVideoCodecCtx->width;
	m_height = m_pVideoCodecCtx->height;
	m_rate = m_pVideoCodecCtx->framerate.num / m_pVideoCodecCtx->framerate.den;
	close();

	return 0;
}

void SlotInfo::close()
{
	if (m_pFormatCtx)
	{
		avformat_close_input(&m_pFormatCtx);
		m_pFormatCtx = NULL;
	}
}