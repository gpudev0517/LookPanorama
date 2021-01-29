#include "CaptureDShow.h"
#include <QDebug>
#include "define.h"
#include "CaptureProp.h"
#include "QmlMainWindow.h"

extern QmlMainWindow* g_mainWindow;

void AVFrameBuffer::alloc(std::size_t cap)
{
	AVFrame* frame = av_frame_alloc();
	buffer.push_back(frame);
}

void AVFrameBuffer::clear()
{
	for (int i = 0; i < buffer.size(); i++)
		av_frame_free(&buffer[i]);
	buffer.clear();
}

AVFrame* AVFrameBuffer::getNext()
{
	if (last >= buffer.size()) last = 0;
	auto p = buffer[last];
	++last;
	if (last == first)
		first = (first + 1) % buffer.size();
	return p;
}

AVFrame* AVFrameBuffer::getFirst()
{
	auto p = first % buffer.size();
	first++;
	return buffer[p];
}

void CaptureDShow::init()
{
	m_cDeviceName = "Camera";
	m_aDeviceName = "";

	m_pVideoCodecCtx = 0;
	m_pAudioFrame = 0;
	m_pTransferAudioFrame = 0;
	m_pFrameRGB = 0;
	m_pFrameBuffer = 0;
	img_convert_ctx = 0;

	m_pAudioCodecCtx = 0;

	m_nWidth = 0;
	m_nHeight = 0;
	m_isAudioDisabled = false;
	m_isLeft = true;
	m_isRight = true;

	av_register_all();
	avdevice_register_all();
	avcodec_register_all();
	m_pFormatCtx = avformat_alloc_context();

	return;
}

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

void CaptureDShow::reset(ImageBufferData& frame)
{
	
}

bool CaptureDShow::open(int index, QString name,
	int width, int height, float fps, //only available in live mode
	Capture::CaptureDomain captureType)
{
	m_nCameraIndex = index;
	m_fps = fps;
	m_captureDomain = captureType;

	bool isDShow = (m_captureDomain == CAPTURE_DSHOW);

	int dupIndex = 0;
	// ffmpeg -list_devices true -f dshow -i dummy
	//ffmpeg -f dshow -list_options true -i video = "USB2.0 Camera"
	QString cDeviceName;
	QString fullDName;
	AVInputFormat *fmt = NULL;
	
	if (isDShow)
	{
		QString deviceName = g_mainWindow->getDeviceNameByDevicePath(name, dupIndex);
		fullDName = cDeviceName.prepend("video=") + ":" + deviceName;
		fmt = av_find_input_format("dshow");
	}
	else
		fullDName = name;
	PANO_LOG("Path of DirectShow device: " + fullDName);

	int ret, i;
	m_nVideoStream = -1;
	m_nAudioStream = -1;
	m_nTotalFrame = 0;

	AVDictionary        *tOptions = NULL;
	if (isDShow)
	{
		char fpsString[10], videosizeString[20];
		sprintf(fpsString, "%d", (int)fps);
		sprintf(videosizeString, "%dx%d", width, height); //"1280x720"

		//av_dict_set(&tOptions, "rtbufsize", "100000000", 0);
		av_dict_set(&tOptions, "framerate", fpsString, 0);
		av_dict_set(&tOptions, "video_size", videosizeString, NULL);
		av_dict_set(&tOptions, "video_device_number", QString::number(dupIndex).toLocal8Bit().data(), 0);
	}

	if (avformat_open_input(&m_pFormatCtx, fullDName.toUtf8().data(), fmt, &tOptions) < 0)
	{
		if (isDShow)
		{
			tOptions = NULL;
			av_dict_set(&tOptions, "video_device_number", QString::number(dupIndex).toLocal8Bit().data(), 0);
			if (avformat_open_input(&m_pFormatCtx, fullDName.toUtf8().data(), fmt, &tOptions) < 0)
			{
				fprintf(stderr, "Could not open input file '%s'", fullDName.toUtf8().data());
				return false;
			}
		}
		else
		{
			fprintf(stderr, "Could not open input file '%s'", fullDName.toUtf8().data());
			return false;
		}
	}
	if ((ret = avformat_find_stream_info(m_pFormatCtx, 0)) < 0)
	{
		fprintf(stderr, "Failed to retrieve input stream information");
		return false;
	}

	if (open_codec_context(&m_nVideoStream, m_pFormatCtx, AVMEDIA_TYPE_VIDEO) >= 0)
	{
		m_pVideoCodecCtx = m_pFormatCtx->streams[m_nVideoStream]->codec;

		m_nWidth = m_pVideoCodecCtx->width;
		m_nHeight = m_pVideoCodecCtx->height;
		
		CameraInput& cameraInput = m_sharedImageBuffer->getGlobalAnimSettings()->getCameraInput(m_nCameraIndex);
		{
			cameraInput.xres = m_nWidth;
			cameraInput.yres = m_nHeight;
			cameraInput.fps = fps;
			cameraInput.pixel_format = m_pVideoCodecCtx->pix_fmt;
		}

		m_videoFrameBuffer.alloc(5);
		m_pFrameRGB = av_frame_alloc();
		

		int num_bytes = avpicture_get_size(AV_PIX_FMT_RGB24, m_nWidth, m_nHeight);
		m_pFrameBuffer = (uint8_t *)av_malloc(num_bytes);
		avpicture_fill((AVPicture*)m_pFrameRGB, m_pFrameBuffer, AV_PIX_FMT_RGB24, m_nWidth, m_nHeight);

		img_convert_ctx = sws_getCachedContext(NULL, m_pVideoCodecCtx->width, m_pVideoCodecCtx->height, m_pVideoCodecCtx->pix_fmt, m_pVideoCodecCtx->width, m_pVideoCodecCtx->height, AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
	}
	else
		return false;

	if (!isDShow && open_codec_context(&m_nAudioStream, m_pFormatCtx, AVMEDIA_TYPE_AUDIO) >= 0)
	{
		m_pAudioCodecCtx = m_pFormatCtx->streams[m_nAudioStream]->codec;
		m_pAudioFrame = av_frame_alloc();
		m_pTransferAudioFrame = av_frame_alloc();
	}

	av_log_set_level(AV_LOG_QUIET);
	//av_log_set_level(AV_LOG_DEBUG);
	av_dump_format(m_pFormatCtx, 0, fullDName.toUtf8().data(), 0);

	//av_log_set_level(AV_LOG_QUIET);

	m_nTotalFrame = (float)(m_pFormatCtx->duration) / (float)(AV_TIME_BASE) * fps - 7;

	return true;
}

int CaptureDShow::decode_packet(AVPacket pkt, int *got_frame)
{
	int ret = 0;
	int decoded = pkt.size;

	*got_frame = 0;

	if (pkt.stream_index == m_nVideoStream) {
		//av_frame_unref(m_pVideoFrame);
		/* decode video frame */
		ret = avcodec_decode_video2(m_pVideoCodecCtx, m_videoFrameBuffer.getNext(), got_frame, &pkt);
		if (ret < 0) {
			//fprintf(stderr, "Error decoding video frame (%s)\n", av_err2str(ret));
			return ret;
		}

		if (*got_frame) {
			//if (m_nCameraIndex == 0)
				//printf("Video[%d]: \t%d\n", this->m_nCameraIndex, m_pVideoFrame->pkt_pts / m_pVideoFrame->pkt_duration);
			m_incomingType = D360::Capture::Video;
		}
	}
	else if (pkt.stream_index == m_nAudioStream) {
		/* decode audio frame */
		ret = avcodec_decode_audio4(m_pAudioCodecCtx, m_pAudioFrame, got_frame, &pkt);
		if (ret < 0) {
			//fprintf(stderr, "Error decoding audio frame (%s)\n", av_err2str(ret));
			return ret;
		}
		/* Some audio decoders decode only part of the packet, and have to be
		* called again with the remainder of the packet data.
		* Sample: fate-suite/lossless-audio/luckynight-partial.shn
		* Also, some decoders might over-read the packet. */
		decoded = FFMIN(ret, pkt.size);

		if (*got_frame && m_isAudioDisabled == false) {
			//printf("Audio[%d]: \t\t\t%d\n", this->m_nCameraIndex, m_pAudioFrame->pkt_pts / m_pAudioFrame->pkt_duration);
			m_incomingType = D360::Capture::Audio;
		}
	}

	return decoded;
}

bool CaptureDShow::grabFrame(ImageBufferData& frame)
{
	int res;
	int frameFinished = 0;
	AVPacket pkt;
	pkt.size = 0;

	int ret = 0;
	int got_frame = 0;

	do
	{
		if (pkt.size == 0)
		{
			pkt.stream_index = -1;
			ret = av_read_frame(m_pFormatCtx, &pkt);
		}

		if (pkt.size == 0) // file read complete
		{
			m_incomingType = D360::Capture::None;
			frameFinished = 1;
			break;
		}

		got_frame = 0;
		ret = decode_packet(pkt, &got_frame);
		if (ret < 0)
			break;
		pkt.data += ret;
		pkt.size -= ret;
		if (pkt.size == 0)
		{
			av_free_packet(&pkt);
		}
	} while (got_frame == 0);

	return frameFinished != 1;
}

bool CaptureDShow::retrieveFrame(int channel, ImageBufferData& frame)
{
	if (m_incomingType == IncomingFrameType::Video)
	{
		AVFrame* videoFrame = m_videoFrameBuffer.getFirst();
		AVPicture* pict = ((AVPicture*)videoFrame);

		if (m_pVideoCodecCtx->pix_fmt == AV_PIX_FMT_YUV420P || m_pVideoCodecCtx->pix_fmt == AV_PIX_FMT_YUVJ420P)
		{
			// yuv420 --> color buffer

			// ==== width ====
			//	 Y		   Y
			//   Y		   Y
			//   U         V

			frame.mFormat = ImageBufferData::YUV420;

			frame.mImageY.buffer = pict->data[0];
			frame.mImageY.width = videoFrame->width;
			frame.mImageY.height = videoFrame->height;
			frame.mImageY.stride = pict->linesize[0];

			frame.mImageU.buffer = pict->data[1];
			frame.mImageU.width = videoFrame->width / 2;
			frame.mImageU.height = videoFrame->height / 2;
			frame.mImageU.stride = pict->linesize[1];

			frame.mImageV.buffer = pict->data[2];
			frame.mImageV.width = videoFrame->width / 2;
			frame.mImageV.height = videoFrame->height / 2;
			frame.mImageV.stride = pict->linesize[2];
		}
#ifdef USE_GPU_COLORCVT
		else if (m_pVideoCodecCtx->pix_fmt == AV_PIX_FMT_YUYV422)
		{
			// yuv422 --> color buffer

			// ==== width ====
			//	 Y		   Y
			//   Y		   Y
			//   U         V
			//   U		   V

			frame.mFormat = ImageBufferData::YUV422;
			
			frame.mImageY.width = videoFrame->width;
			frame.mImageY.height = videoFrame->height;
			frame.mImageY.stride = pict->linesize[0];
			if (frame.mImageY.buffer == NULL)
			{
				frame.mImageY.makeBuffer(frame.mImageY.stride * frame.mImageY.height);
			}
			memcpy(frame.mImageY.buffer, pict->data[0], frame.mImageY.stride * frame.mImageY.height);
			frame.mImageU = AlignedImage();
			frame.mImageV = AlignedImage();
		}
		else if (m_pVideoCodecCtx->pix_fmt == AV_PIX_FMT_YUVJ422P)
		{
			// yuv422 --> color buffer

			// ==== width ====
			//	 Y		   Y
			//   Y		   Y
			//   U         V
			//   U		   V

			frame.mFormat = ImageBufferData::YUVJ422P;

			frame.mImageY.buffer = pict->data[0];
			frame.mImageY.width = videoFrame->width;
			frame.mImageY.height = videoFrame->height;
			frame.mImageY.stride = pict->linesize[0];

			frame.mImageU.buffer = pict->data[1];
			frame.mImageU.width = videoFrame->width / 2;
			frame.mImageU.height = videoFrame->height;
			frame.mImageU.stride = pict->linesize[1];

			frame.mImageV.buffer = pict->data[2];
			frame.mImageV.width = videoFrame->width / 2;
			frame.mImageV.height = videoFrame->height;
			frame.mImageV.stride = pict->linesize[2];
		}
		else if (m_pVideoCodecCtx->pix_fmt == AV_PIX_FMT_BGR0)
		{
			// bgr0 --> color buffer

			frame.mFormat = ImageBufferData::BGR0;

			frame.mImageY.width = videoFrame->width;
			frame.mImageY.height = videoFrame->height;
			frame.mImageY.stride = videoFrame->width * 4;
			frame.mImageY.buffer = pict->data[0] - frame.mImageY.stride * (frame.mImageY.height - 1);
			frame.mImageU = AlignedImage();
			frame.mImageV = AlignedImage();
		}
#endif
		else
		{
			sws_scale(img_convert_ctx, pict->data, pict->linesize, 0, m_pVideoCodecCtx->height, pict->data, ((AVPicture *)m_pFrameRGB)->linesize);
			//QImage snapshotImage(m_pFrameRGB->data[0], m_pVideoFrame->width, m_pVideoFrame->height, QImage::Format::Format_RGB888);
			frame.mFormat = ImageBufferData::RGB888;
			frame.mImageY.buffer = m_pFrameRGB->data[0];
			frame.mImageY.width = videoFrame->width;
			frame.mImageY.height = videoFrame->height;
			frame.mImageY.stride = videoFrame->width * 3;
			frame.mImageU = AlignedImage();
			frame.mImageV = AlignedImage();
		}

		m_incomingType = IncomingFrameType::None;
#if 0	// Original snapshot function
		if (isSnapshot) // screen capture
		{
			sws_scale(img_convert_ctx, ((AVPicture*)m_pVideoFrame)->data, ((AVPicture*)m_pVideoFrame)->linesize, 0, m_pVideoCodecCtx->height, ((AVPicture *)m_pFrameRGB)->data, ((AVPicture *)m_pFrameRGB)->linesize);
			QImage snapshotImage(m_pFrameRGB->data[0], m_pVideoFrame->width, m_pVideoFrame->height, QImage::Format::Format_RGB888);

			snapshotImage.save(m_snapshot + QString("%1.png").arg(m_nCameraIndex));

			isSnapshot = false;
		}
#endif
		if (frame.mImageY.width == 0 || frame.mImageY.height == 0)
		{
			PANO_WARN_ARG("Invalid frame captured: [%1]", this->m_nCameraIndex);
			return false;
		}	
		return true;
	}

	return false;
}

void CaptureDShow::seekFirst()
{
	av_seek_frame(m_pFormatCtx, -1, 0, AVSEEK_FLAG_ANY);
	/*//printf("**** seekFrame to %d. LLT: %d. LT: %d. LLF: %d. LF: %d. LastFrameOk: %d\n",(int)frame,LastLastFrameTime,LastFrameTime,LastLastFrameNumber,LastFrameNumber,(int)LastFrameOk);

	// Seek if:
	// - we don't know where we are (Ok=false)
	// - we know where we are but:
	//    - the desired frame is after the last decoded frame (this could be optimized: if the distance is small, calling decodeSeekFrame may be faster than seeking from the last key frame)
	//    - the desired frame is smaller or equal than the previous to the last decoded frame. Equal because if frame==LastLastFrameNumber we don't want the LastFrame, but the one before->we need to seek there
	if ((LastFrameOk == false) || ((LastFrameOk == true) && (frame <= LastLastFrameNumber || frame > LastFrameNumber)))
	{
		//printf("\t avformat_seek_file\n");
		if (ffmpeg::avformat_seek_file(pFormatCtx, videoStream, 0, frame, frame, AVSEEK_FLAG_FRAME) < 0)
			return false;

		avcodec_flush_buffers(pCodecCtx);

		DesiredFrameNumber = frame;
		LastFrameOk = false;
	}
	//printf("\t decodeSeekFrame\n");

	return decodeSeekFrame(frame);*/
}

void CaptureDShow::seekFrames(int nFrames)
{
// 	int64_t timeBase = (int64_t(m_pVideoCodecCtx->time_base.den)) / int64_t(m_pVideoCodecCtx->time_base.num);
// 	int seekTarget = ((float)(nFrames) / m_fps) * timeBase;
// 
// 	int nRet = av_seek_frame(m_pFormatCtx, m_nVideoStream, seekTarget, AVSEEK_FLAG_ANY);
// 	if (nRet < 0)
// 		PANO_LOG("av_seek_frame failed.");

	int newFrames = nFrames;
	int curFrames = m_pVideoCodecCtx->frame_number;
	if (curFrames > 1)
	{
		av_seek_frame(m_pFormatCtx, m_nVideoStream, curFrames, AVSEEK_FLAG_BACKWARD);
		m_pVideoCodecCtx->frame_number = 1;
		newFrames += 7;
	}

	if (newFrames == 0)
	{
		newFrames++;
	}

	while (curFrames != newFrames)
	{
		AVPacket packet;
		int frame_done;

		while (av_read_frame(m_pFormatCtx, &packet) >= 0)
		{
			avcodec_decode_video2(m_pVideoCodecCtx, m_videoFrameBuffer.getNext(), &frame_done, &packet);
			if (frame_done) {
				curFrames = m_pVideoCodecCtx->frame_number;
				av_free_packet(&packet);
				break;
			}
			av_free_packet(&packet);
		}
	}
}

ImageBufferData CaptureDShow::convertToRGB888(ImageBufferData& image)
{
	SwsContext* convert_ctx = NULL;
	AVPixelFormat pix_fmt = AV_PIX_FMT_YUV420P;
	const uint8_t* inData[8] = {NULL};
	int inLineSize[8] = {0};

	switch (image.mFormat)
	{
	case ImageBufferData::YUV420: 
		pix_fmt = AV_PIX_FMT_YUV420P; 
		inData[0] = image.mImageY.buffer;
		inLineSize[0] = image.mImageY.stride;
		inData[1] = image.mImageU.buffer;
		inLineSize[1] = image.mImageU.stride;
		inData[2] = image.mImageV.buffer;
		inLineSize[2] = image.mImageV.stride;
		break;
	case ImageBufferData::YUV422: 
		pix_fmt = AV_PIX_FMT_YUV422P; 
		inData[0] = image.mImageY.buffer;
		inLineSize[0] = image.mImageY.stride;
		break;
	default:	// ImageBufferData::RBB888;
		return image;
	}
	
	
	int newHeight = sws_scale(img_convert_ctx, inData, inLineSize, 0, image.mImageY.height, m_pFrameRGB->data, m_pFrameRGB->linesize);

	ImageBufferData outImage(ImageBufferData::RGB888);
	outImage.mImageY.buffer = m_pFrameRGB->data[0];
	outImage.mImageY.width = image.mImageY.width;
	outImage.mImageY.height = image.mImageY.height;
	outImage.mImageY.stride = m_pFrameRGB->linesize[0];

	return outImage;
}

void* CaptureDShow::retrieveAudioFrame()
{
	if (m_incomingType != IncomingFrameType::Audio)		return NULL;

	//m_pTransferAudioFrame = m_pAudioFrame;
	m_pTransferAudioFrame->format = m_pAudioFrame->format;
	m_pTransferAudioFrame->sample_rate = m_pAudioFrame->sample_rate;
	m_pTransferAudioFrame->nb_samples = m_pAudioFrame->nb_samples;
	if (m_pAudioFrame->channels == 0)
		m_pTransferAudioFrame->channels = av_get_channel_layout_nb_channels(m_pAudioFrame->channel_layout);
	else
		m_pTransferAudioFrame->channels = m_pAudioFrame->channels;
	m_pTransferAudioFrame->channels = m_isLeft & m_isRight ? m_pTransferAudioFrame->channels : 1;
	m_pTransferAudioFrame->channel_layout = av_get_default_channel_layout(m_pTransferAudioFrame->channels);

	av_frame_get_buffer(m_pTransferAudioFrame, 0);
	if (m_isLeft & m_isRight) {
		memcpy(m_pTransferAudioFrame->extended_data[0], m_pAudioFrame->extended_data[0], m_pAudioFrame->nb_samples * av_get_bytes_per_sample(m_pAudioCodecCtx->sample_fmt));
		memcpy(m_pTransferAudioFrame->extended_data[1], m_pAudioFrame->extended_data[1], m_pAudioFrame->nb_samples * av_get_bytes_per_sample(m_pAudioCodecCtx->sample_fmt));
	}
	else if (m_isLeft)
		memcpy(m_pTransferAudioFrame->extended_data[0], m_pAudioFrame->extended_data[0], m_pAudioFrame->nb_samples * av_get_bytes_per_sample(m_pAudioCodecCtx->sample_fmt));
	else if (m_isRight)
		memcpy(m_pTransferAudioFrame->extended_data[0], m_pAudioFrame->extended_data[1], m_pAudioFrame->nb_samples * av_get_bytes_per_sample(m_pAudioCodecCtx->sample_fmt));

	m_incomingType = IncomingFrameType::None;

	return m_pTransferAudioFrame;
}

void CaptureDShow::close()
{
	//avcodec_close(m_pCodecCtx);
	m_videoFrameBuffer.clear();
	if (m_pAudioFrame) av_frame_free(&m_pAudioFrame);
	if (m_pFrameRGB) av_frame_free(&m_pFrameRGB);
	if (m_pFrameBuffer)
	{
		av_free(m_pFrameBuffer);
		m_pFrameBuffer = NULL;
	}
	//if (m_pFrameBuffer)	av_freep(m_pFrameBuffer);
	if (m_pTransferAudioFrame) av_frame_free(&m_pTransferAudioFrame);
	if (img_convert_ctx)
	{
		sws_freeContext(img_convert_ctx);
		img_convert_ctx = 0;
	}

	if (m_pVideoCodecCtx)
	{
		avcodec_close(m_pVideoCodecCtx);
		m_pVideoCodecCtx = NULL;
	}
	if (m_pAudioCodecCtx)
	{
		avcodec_close(m_pAudioCodecCtx);
		m_pAudioCodecCtx = NULL;
	}

	if (m_pFormatCtx)
	{
		avformat_close_input(&m_pFormatCtx);
	}
}

double CaptureDShow::getProperty(int property_id)
{
	if (m_pVideoCodecCtx == NULL)
		return 0;

	int ival = 0;
	float fval = 0;

	switch (property_id)
	{
		// OCV parameters
	case CV_CAP_PROP_FRAME_WIDTH: return m_pVideoCodecCtx->width;
	case CV_CAP_PROP_FRAME_HEIGHT: return m_pVideoCodecCtx->height;
	}

	return 0.0f;
}

// -1:Disabled, 0:Mixed, 1:Left, 2:Right
void CaptureDShow::setAudio(CameraInput::InputAudioChannelType audioType)
{
	if (audioType == CameraInput::NoAudio) {	// Disabled
		m_isAudioDisabled = true;
		m_isLeft = false;	
		m_isRight = false;
		return;
	}

	if (audioType == CameraInput::MixedChannel) {	// Mixed
		m_isLeft = true;
		m_isRight = false;
	}
	else if (audioType == CameraInput::LeftChannel) {	// Left
		m_isLeft = true;
		m_isRight = false;
	}
	else if (audioType == CameraInput::RightChannel) {	// Right
		m_isLeft = false;
		m_isRight = true;
	}
}