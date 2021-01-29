#include <iostream>



#include "StreamFfmpeg.hpp"
#include "FfmpegUtils.hpp"

extern "C"
{
#include "libavutil/avassert.h"
#include "libavutil/timestamp.h"
}


#include "CaptureAudio.h"
#include "TLogger.h"
#include "define.h"


/**
* Initialize a temporary storage for the specified number of audio samples.
* The conversion requires temporary storage due to the different format.
* The number of audio samples to be allocated is specified in frame_size.
*/
int init_converted_samples(uint8_t ***converted_input_samples,
	AVCodecContext *output_codec_context,
	int frame_size)
{
	int error;

	/**
	* Allocate as many pointers as there are audio channels.
	* Each pointer will later point to the audio samples of the corresponding
	* channels (although it may be NULL for interleaved formats).
	*/
	if (!(*converted_input_samples = (uint8_t**)calloc(output_codec_context->channels,
		sizeof(**converted_input_samples)))) {
		fprintf(stderr, "Could not allocate converted input sample pointers\n");
		return AVERROR(ENOMEM);
	}

	/**
	* Allocate memory for the samples of all channels in one consecutive
	* block for convenience.
	*/
	if ((error = av_samples_alloc(*converted_input_samples, NULL,
		output_codec_context->channels,
		frame_size,
		output_codec_context->sample_fmt, 0)) < 0) {
		fprintf(stderr,
			"Could not allocate converted input samples (error '%s')\n",
			get_error_text(error));
		av_freep(&(*converted_input_samples)[0]);
		free(*converted_input_samples);
		return error;
	}
	return 0;
}

/**
* Convert the input audio samples into the output sample format.
* The conversion happens on a per-frame basis, the size of which is specified
* by frame_size.
*/
int convert_samples(const uint8_t **input_data,
	uint8_t **converted_data, const int frame_size,
	SwrContext *resample_context)
{
	int error = 0;

	try
	{
		/** Convert the samples using the resampler. */
		if ((error = swr_convert(resample_context,
			converted_data, frame_size,
			input_data, frame_size)) < 0)
		{
			fprintf(stderr, "Could not convert input samples (error '%s')\n",
				get_error_text(error));
			return error;
		}
	}
	catch (...)
	{
		fprintf(stderr, "swr_convert crashed for audio frame\n");
		error = 1;
	}

	return 0;
}

/** Add converted input audio samples to the FIFO buffer for later processing. */
int add_samples_to_fifo(AVAudioFifo *fifo,
	uint8_t **converted_input_samples,
	const int frame_size)
{
	int error;

	/**
	* Make the FIFO as large as it needs to be to hold both,
	* the old and the new samples.
	*/
	if ((error = av_audio_fifo_realloc(fifo, av_audio_fifo_size(fifo) + frame_size)) < 0) {
		fprintf(stderr, "Could not reallocate FIFO\n");
		return error;
	}

	/** Store the new samples in the FIFO buffer. */
	if (av_audio_fifo_write(fifo, (void **)converted_input_samples,
		frame_size) < frame_size) {
		fprintf(stderr, "Could not write data to FIFO\n");
		return AVERROR_EXIT;
	}
	return 0;
}

/**
* Initialize one input frame for writing to the output file.
* The frame will be exactly frame_size samples large.
*/
int init_output_frame(AVFrame **frame,
	AVCodecContext *output_codec_context,
	int frame_size)
{
	int error;

	/** Create a new frame to store the audio samples. */
	if (!(*frame = av_frame_alloc())) {
		fprintf(stderr, "Could not allocate output frame\n");
		return AVERROR_EXIT;
	}

	/**
	* Set the frame's parameters, especially its size and format.
	* av_frame_get_buffer needs this to allocate memory for the
	* audio samples of the frame.
	* Default channel layouts based on the number of channels
	* are assumed for simplicity.
	*/
	(*frame)->nb_samples = frame_size;
	(*frame)->channel_layout = output_codec_context->channel_layout;
	(*frame)->format = output_codec_context->sample_fmt;
	(*frame)->sample_rate = output_codec_context->sample_rate;

	/**
	* Allocate the samples of the created frame. This call will make
	* sure that the audio frame can hold as many samples as specified.
	*/
	if ((error = av_frame_get_buffer(*frame, 0)) < 0) {
		fprintf(stderr, "Couldn't allocate output frame samples (error '%s')\n",
			get_error_text(error));
		av_frame_free(frame);
		return error;
	}

	return 0;
}


StreamFfmpeg::StreamFfmpeg()
{
	Clean();
}

void StreamFfmpeg::Clean()
{
	videoIndex = -1;
	frameIndex = 0;
	startTime = 0;
	iFormatContext_ = NULL;
	formatContext_ = NULL;
	errVal = 0;

	mHaveAudio = 0;
	mHaveVideo = 0;
	mEncodeVideo = false;
	mEncodeAudio = false;
	m_inSampleFmt = AV_SAMPLE_FMT_FLTP;

	m_capType = D360::Capture::CaptureDomain::CAPTURE_VIDEO;

	mOutputVideoStream.Initialize();
	mOutputAudioStream.Initialize();

	mInitialized = false;
}

//////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////



int StreamFfmpeg::Initialize(std::string outFileName, int width, int height, int fps, int channelCount,
	AVSampleFormat sampleFmt, int srcSampleRate, int sampleRate, int audioLag, bool toFile, int videoCodec, int audioCodec, bool bLidar, int crf)
{
	streamThread.lock();
	pts = 0;

	int ret = -1;
	mInitialized = false;

	this->audioLag = audioLag;
	m_inSampleFmt = sampleFmt;

	int channels = channelCount > AV_NUM_DATA_POINTERS ? AV_NUM_DATA_POINTERS : channelCount;
	if (channels == 7)	channels = 6;	// In case that, error opened.
	//if (m_inSampleFmt <= AV_SAMPLE_FMT_U8P && channels >= 2)
	//	m_inSampleFmt = (AVSampleFormat)(m_inSampleFmt + 5);		// Convert to planar format

	//av_log_set_level(AV_LOG_VERBOSE);

	av_register_all();
	//
	// Network
	//

	avformat_network_init();

	//
	// Input AVFormatContext and Output AVFormatContext
	//
	const char *out_filename = outFileName.c_str();

	
	//AVOutputFormat *fmt = NULL;
	//while ((fmt = av_oformat_next(fmt))) {
	//	std::cout << fmt->name << "     " << fmt->long_name << std::endl;
	//}

	//
	// Allocate the output media context 
	//
	if (outFileName.substr(0, 4) == "rtsp")
		avformat_alloc_output_context2(&formatContext_, NULL, "rtsp", out_filename);
	else if (outFileName.substr(0, 4) == "rtmp")
		avformat_alloc_output_context2(&formatContext_, NULL, "flv", out_filename);
	else if (outFileName.substr(0, 3) == "sdp")
		avformat_alloc_output_context2(&formatContext_, NULL, "flv", out_filename);
	else
		avformat_alloc_output_context2(&formatContext_, NULL, "flv", out_filename);
	if( !formatContext_ ) 
	{
		std::cerr << "Could not deduce output format from file extension : using MPEG. " << std::endl;
		streamThread.unlock();
		return -1;
	}


	outputFormat_ = formatContext_->oformat;

	// Add the audio and video streams using the default format codecs
	// and initialize the codecs. 
	//

	AVDictionary *opt = NULL;

	if( outputFormat_->video_codec != AV_CODEC_ID_NONE ) 
	{
		videoFps = fps;
		outputFormat_->video_codec = (enum AVCodecID)videoCodec;
		ret = add_stream(&mOutputVideoStream, formatContext_, &mVideoCodec, outputFormat_->video_codec,
			width, height, fps, toFile, sampleRate, channels, bLidar);
		if (ret < 0)
		{
			streamThread.unlock();
			return ret;
		}
		mHaveVideo		= 1;
		mEncodeVideo	= true;
	}
	if (outputFormat_->audio_codec != AV_CODEC_ID_NONE && channels != 0)
	{	
		outputFormat_->audio_codec = AV_CODEC_ID_AAC; //AV_CODEC_ID_MP3
		ret = add_stream(&mOutputAudioStream, formatContext_, &mAudioCodec, outputFormat_->audio_codec,
			width, height, fps, toFile, sampleRate, channels, bLidar);
		if (ret < 0)
		{
			streamThread.unlock();
			return ret;
		}	
		mHaveAudio		= 1;
		mEncodeAudio	= 1;

		Init_Audio_Resampler(channels, m_inSampleFmt, srcSampleRate);
		Init_Audio_Fifo();
	}

	//
	// Now that all the parameters are set, we can open the audio and
	// video codec s and allocate the necessary encode buffers. 
	//
	if (mHaveVideo)
	{
		// include / x264.h:static const char * const x264_preset_names[] = { "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo", 0 };
		// "profile", { "main", "high", "high444" }
		// "tune", { film, animation, grain, stillimage, psnr, ssim, fastdecode, zerolatency}
		if (outputFormat_->video_codec == AV_CODEC_ID_H264 ) {
			if (toFile)
			{
				// loss-less
				av_dict_set(&opt, "preset", "ultrafast", 0);
				//av_dict_set(&opt, "profile", "baseline", 0);
				// To ensure compatibility, you should always therefore add
				// the option -pix_fmt yuv420p and choose a lossy encoding method,
				// ideally by using CRF encoding, and setting -crf to a reasonable value.
				// CRF default is 23, but anything from 18–28 is fine, where lower means better quality.
				av_dict_set(&opt, "crf", QString::number(crf).toLocal8Bit().data(), 0);
				//av_dict_set(&opt, "crf", "0", 0);
				//av_dict_set(&opt, "qp", "0", 0);
				av_dict_set(&opt, "tune", "film", 0);
			}
			else
			{
				av_dict_set(&opt, "preset", "ultrafast", 0);
				av_dict_set(&opt, "profile", "main", 0);
			}
		}
		else if (outputFormat_->video_codec == AV_CODEC_ID_H265)
		{
			if (toFile)
			{
				// "preset", { "hp", "hq", "bd", "ll", "llhp", "llhq", "lossless", "losslesshp", "default" }
				//av_dict_set(&opt, "preset", "fast", 0);

				av_dict_set(&opt, "x265-params", "qp=20", 0);
				av_dict_set(&opt, "preset", "ultrafast", 0);
				av_dict_set(&opt, "tune", "zero-latency", 0);
				av_dict_set(&opt, "qmin", "0", 0);
				av_dict_set(&opt, "qmax", "69", 0);
				av_dict_set(&opt, "qdiff", "4", 0);
			}
		}	
		ret = open_video(formatContext_, mVideoCodec, &mOutputVideoStream, opt);
		if (ret < 0)
		{
			streamThread.unlock();
			return ret;
		}
	}

	if (mHaveAudio)
	{
		opt = 0;
		ret = open_audio(formatContext_, mAudioCodec, &mOutputAudioStream, opt);
	}

	av_dump_format( formatContext_, 1, out_filename, 1 );

	//av_log_set_level(AV_LOG_DEBUG);
	//
	// open the output file, if needed 
	//
	if( !(outputFormat_->flags & AVFMT_NOFILE ) ) 
	{
		int ret = avio_open( &formatContext_->pb, out_filename, AVIO_FLAG_WRITE);
		//av_log_set_level(AV_LOG_WARNING);
		if( ret < 0 ) 
		{
			char errString[100];
			sprintf(errString, "Could not open %s", out_filename);
			T_LOG(errString);
			//std::cerr << "Could not open " << out_filename << std::endl;
			streamThread.unlock();
			return ret;
		}
	}

	//
	// Write the stream header, if any. 
	//
	ret = avformat_write_header( formatContext_, &opt );
	//av_log_set_level(AV_LOG_WARNING);
	if( ret < 0 ) 
	{
		T_WARNING("Error occurred when opening output file: ");
		streamThread.unlock();
		return ret;
	}

	mInitialized = true;
	streamThread.unlock();
	return 1;
}

int StreamFfmpeg::Initialize_webRTC(std::string outFileName, int width, int height, int fps, int channelCount, AVSampleFormat sampleFmt,
	int srcSampleRate, int sampleRate, int audioLag, bool toFile, int videoCodec, int audioCodec, int crf)
{
	streamThread.lock();
	pts = 0;

	int ret = -1;
	mInitialized = false;

	this->audioLag = audioLag;
	m_inSampleFmt = sampleFmt;

	int channels = channelCount > AV_NUM_DATA_POINTERS ? AV_NUM_DATA_POINTERS : channelCount;
	if (channels == 7)	channels = 6;	// In case that, error opened.
										//if (m_inSampleFmt <= AV_SAMPLE_FMT_U8P && channels >= 2)
										//	m_inSampleFmt = (AVSampleFormat)(m_inSampleFmt + 5);		// Convert to planar format

										//av_log_set_level(AV_LOG_VERBOSE);

	av_register_all();
	//
	// Network
	//

	avformat_network_init();

	//
	// Input AVFormatContext and Output AVFormatContext
	//
	const char *out_filename = outFileName.c_str();


	//AVOutputFormat *fmt = NULL;
	//while ((fmt = av_oformat_next(fmt))) {
	//	std::cout << fmt->name << "     " << fmt->long_name << std::endl;
	//}

	//
	// Allocate the output media context 
	//
	avformat_alloc_output_context2(&formatContext_, NULL, "flv", out_filename);
	//avformat_alloc_output_context2(&formatContext_, NULL, "h264", out_filename);
	if (!formatContext_)
	{
		std::cerr << "Could not deduce output format from file extension : using MPEG. " << std::endl;
		streamThread.unlock();
		return -1;
	}

	outputFormat_ = formatContext_->oformat;

	// Add the audio and video streams using the default format codecs
	// and initialize the codecs. 
	//

	AVDictionary *opt = NULL;

	if (outputFormat_->video_codec != AV_CODEC_ID_NONE)
	{
		videoFps = fps;
		outputFormat_->video_codec = (enum AVCodecID)videoCodec;
		ret = add_stream(&mOutputVideoStream, formatContext_, &mVideoCodec, outputFormat_->video_codec,
			width, height, fps, toFile, sampleRate, channels, false);
		if (ret < 0)
		{
			streamThread.unlock();
			return ret;
		}
		mHaveVideo = 1;
		mEncodeVideo = true;
	}
	if (outputFormat_->audio_codec != AV_CODEC_ID_NONE && channels != 0)
	{
		outputFormat_->audio_codec = AV_CODEC_ID_AAC; //AV_CODEC_ID_MP3
		ret = add_stream(&mOutputAudioStream, formatContext_, &mAudioCodec, outputFormat_->audio_codec,
			width, height, fps, toFile, sampleRate, channels, false);
		if (ret < 0)
		{
			streamThread.unlock();
			return ret;
		}
		mHaveAudio = 1;
		mEncodeAudio = 1;

		Init_Audio_Resampler(channels, m_inSampleFmt, srcSampleRate);
		Init_Audio_Fifo();
	}

	//
	// Now that all the parameters are set, we can open the audio and
	// video codec s and allocate the necessary encode buffers. 
	//
	if (mHaveVideo)
	{
		// include / x264.h:static const char * const x264_preset_names[] = { "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo", 0 };
		// "profile", { "main", "high", "high444" }
		// "tune", { film, animation, grain, stillimage, psnr, ssim, fastdecode, zerolatency}
		if (outputFormat_->video_codec == AV_CODEC_ID_H264) {
			if (toFile)
			{
				// loss-less
				av_dict_set(&opt, "preset", "ultrafast", 0);
				//av_dict_set(&opt, "profile", "baseline", 0);
				// To ensure compatibility, you should always therefore add
				// the option -pix_fmt yuv420p and choose a lossy encoding method,
				// ideally by using CRF encoding, and setting -crf to a reasonable value.
				// CRF default is 23, but anything from 18–28 is fine, where lower means better quality.
				av_dict_set(&opt, "crf", QString::number(crf).toLocal8Bit().data(), 0);
				//av_dict_set(&opt, "crf", "0", 0);
				//av_dict_set(&opt, "qp", "0", 0);
				av_dict_set(&opt, "tune", "film", 0);
			}
			else
			{
				av_dict_set(&opt, "preset", "ultrafast", 0);
				av_dict_set(&opt, "profile", "main", 0);
			}
		}
		else if (outputFormat_->video_codec == AV_CODEC_ID_H265)
		{
			if (toFile)
			{
				// "preset", { "hp", "hq", "bd", "ll", "llhp", "llhq", "lossless", "losslesshp", "default" }
				//av_dict_set(&opt, "preset", "fast", 0);

				av_dict_set(&opt, "x265-params", "qp=20", 0);
				av_dict_set(&opt, "preset", "ultrafast", 0);
				av_dict_set(&opt, "tune", "zero-latency", 0);
				av_dict_set(&opt, "qmin", "0", 0);
				av_dict_set(&opt, "qmax", "69", 0);
				av_dict_set(&opt, "qdiff", "4", 0);
			}
		}

		ret = open_video(formatContext_, mVideoCodec, &mOutputVideoStream, opt);
		if (ret < 0)
		{
			streamThread.unlock();
			return ret;
		}
	}

	if (mHaveAudio)
	{
		opt = 0;
		ret = open_audio(formatContext_, mAudioCodec, &mOutputAudioStream, opt);
	}

	av_dump_format(formatContext_, 1, out_filename, 1);

	mInitialized = true;
	streamThread.unlock();
	return 1;
}

int StreamFfmpeg::Init_Audio_Resampler(int channelCount, AVSampleFormat sampleFmt, int sampleRate)
{
	int error;

	AVCodecContext * output_codec_context = mOutputAudioStream.st->codec;
	/**
	* Create a resampler context for the conversion.
	* Set the conversion parameters.
	* Default channel layouts based on the number of channels
	* are assumed for simplicity (they are sometimes not detected
	* properly by the demuxer and/or decoder).
	*/

	/*resample_context = swr_alloc_set_opts(NULL,
		av_get_default_channel_layout(output_codec_context->channels),
		output_codec_context->sample_fmt,
		output_codec_context->sample_rate,
		av_get_default_channel_layout(channelCount),
		sampleFmt,
		sampleRate,
		0, NULL);
	if (!resample_context) {
		fprintf(stderr, "Could not allocate resample context\n");
		return AVERROR(ENOMEM);
	}*/
	// Set up SWR context once you've got codec information
	resample_context = swr_alloc();

	int in_channel_layout = av_get_default_channel_layout(channelCount);
	int out_channel_layout = av_get_default_channel_layout(output_codec_context->channels);
	av_opt_set_int(resample_context, "in_channel_layout", in_channel_layout, 0);
	av_opt_set_int(resample_context, "out_channel_layout", out_channel_layout, 0);
	av_opt_set_int(resample_context, "in_sample_rate", sampleRate, 0);
	av_opt_set_int(resample_context, "out_sample_rate", output_codec_context->sample_rate, 0);
	av_opt_set_sample_fmt(resample_context, "in_sample_fmt", sampleFmt, 0);
	av_opt_set_sample_fmt(resample_context, "out_sample_fmt", output_codec_context->sample_fmt, 0);
	/**
	* Perform a sanity check so that the number of converted samples is
	* not greater than the number of samples to be converted.
	* If the sample rates differ, this case has to be handled differently
	*/
	//av_assert0(output_codec_context->sample_rate == sampleRate);

	/** Open the resampler with the specified parameters. */
	if ((error = swr_init(resample_context)) < 0) {
		fprintf(stderr, "Could not open resample context\n");
		swr_free(&resample_context);
		return error;
	}
	return 0;
}

int StreamFfmpeg::Init_Audio_Fifo()
{
	AVCodecContext * output_codec_context = mOutputAudioStream.st->codec;
	/** Create the FIFO buffer based on the specified output sample format. */
	if (!(fifo = av_audio_fifo_alloc(output_codec_context->sample_fmt,
		output_codec_context->channels, 1))) {
		fprintf(stderr, "Could not allocate FIFO\n");
		return AVERROR(ENOMEM);
	}
	return 0;
}

int StreamFfmpeg::StreamFrame( uint8_t* input, int frame_index, int width, int height, bool sync )
{
	if (!mInitialized) return 0;
	if (!mHaveAudio) sync = false;

	streamThread.lock();
	//while( encode_video || encode_audio ) 
	{
		//
		// select the stream to encode 
		//
		/*if ( mEncodeVideo &&
			(!mEncodeAudio || av_compare_ts( mOutputVideoStream.next_pts, mOutputVideoStream.st->codec->time_base,
			mOutputAudioStream.next_pts, mOutputAudioStream.st->codec->time_base) <= 0)) */
		{
			//mEncodeVideo = !write_video_frame( formatContext_, &mOutputVideoStream, input, frame_index );
		}

		//if (mEncodeVideo) {
			//mEncodeVideo = !write_video_frame(formatContext_, &mOutputVideoStream, input, frame_index);
		streamMutex.lock();
		//write_video_frame(formatContext_, &mOutputVideoStream, input, frame_index, videoFps, m_capType == D360::Capture::CaptureDomain::CAPTURE_DSHOW);
		write_video_frame(formatContext_, &mOutputVideoStream, input, frame_index, videoFps, sync);
		streamMutex.unlock();
		//}
		//else {
			
		//}
		//StreamAudio();
	}
	streamThread.unlock();


	return 1;
}

int StreamFfmpeg::StreamLiDARFrame( uint8_t* input, int width, int height )
{
	if( !mInitialized ) return 0;

	uint8_t* lidarInput = new uint8_t[LIDAR_STREAM_HEIGHT * LIDAR_STREAM_WIDTH + LIDAR_STREAM_HEIGHT * LIDAR_STREAM_WIDTH / 2];
	memcpy( lidarInput, input, LIDAR_STREAM_HEIGHT * LIDAR_STREAM_WIDTH );
	memset( lidarInput + LIDAR_STREAM_HEIGHT * LIDAR_STREAM_WIDTH, 128, LIDAR_STREAM_HEIGHT* LIDAR_STREAM_WIDTH / 2 );
	streamThread.lock();
	streamMutex.lock();
	write_video_frame( formatContext_, &mOutputVideoStream, lidarInput, 0, videoFps, false );
	streamMutex.unlock();
	streamThread.unlock();
	free( lidarInput );

	return 1;
}

AVFrame* StreamFfmpeg::webRTC_makeStreamFrame(uint8_t* input, int frame_index, int width, int height, bool sync)
{
	if (!mInitialized) return 0;
	if (!mHaveAudio) sync = false;

	streamThread.lock();
	streamMutex.lock();

	//write_video_frame(formatContext_, &mOutputVideoStream, input, frame_index, videoFps, m_capType == D360::Capture::CaptureDomain::CAPTURE_DSHOW);
	AVFrame* retAVFrame = WebRTC_write_video_frame(formatContext_, &mOutputVideoStream, input, frame_index, videoFps, sync);
	
	streamMutex.unlock();
	streamThread.unlock();

	return retAVFrame;
}

int StreamFfmpeg::StoreAudioFrame(QMap<int, void*> audioFrame)
{
#define MAX_CHANNEL_COUNT	AV_NUM_DATA_POINTERS	// Default is 8
	if (audioFrame.size() == 0)		return -1;

	AVFrame* outFrame = av_frame_alloc();
	AVFrame* inFrame = (AVFrame*)audioFrame.first();
	outFrame->format = inFrame->format;
	outFrame->sample_rate = inFrame->sample_rate;
	outFrame->nb_samples = inFrame->nb_samples;
	outFrame->channels = 0;
	QMapIterator<int, void*> iter(audioFrame);
	while (iter.hasNext()) {
		iter.next();
		outFrame->channels += ((AVFrame*)iter.value())->channels;
	}
	
	if (outFrame->channels > MAX_CHANNEL_COUNT)
		outFrame->channels = MAX_CHANNEL_COUNT;
	if (outFrame->channels == 7)	// In case that, error opened!
		outFrame->channels = 6;
	
	//int orgChannels = outFrame->channels;
	//if (outFrame->channels == 1)	// AAC codec must be more than 2 channels
	//	outFrame->channels = 2;

	outFrame->channel_layout = av_get_default_channel_layout(outFrame->channels);
	//if (outFrame->format <= AV_SAMPLE_FMT_U8P) //&& outFrame->channels >= 2
	//	outFrame->format += AV_SAMPLE_FMT_U8P;	// Convert to planar format

	av_frame_get_buffer(outFrame, 0);
	
	iter.toFront();
	int idx = 0;
	while (iter.hasNext()) {
		iter.next();
		inFrame = (AVFrame*)iter.value();

		memcpy(outFrame->extended_data[idx++], inFrame->extended_data[0], inFrame->nb_samples * av_get_bytes_per_sample((AVSampleFormat)inFrame->format));
		if (idx >= outFrame->channels)	break;
		
		if (inFrame->channels > 1)	{	// In case that, this frame is stereo channel.
			memcpy(outFrame->extended_data[idx++], inFrame->extended_data[1], inFrame->nb_samples * av_get_bytes_per_sample((AVSampleFormat)inFrame->format));
			if (idx >= outFrame->channels)	break;
		}

		//if (orgChannels == 1) {
		//	memcpy(outFrame->extended_data[idx], inFrame->extended_data[0], inFrame->nb_samples * av_get_bytes_per_sample((AVSampleFormat)inFrame->format));
		//	break;
		//}
	}
	/*
	AVFrame* inFrame1 = (AVFrame*)audioFrame[1];
	memcpy(outFrame->extended_data[0], inFrame->extended_data[0], inFrame->nb_samples * av_get_bytes_per_sample(AV_SAMPLE_FMT_FLTP));
	memcpy(outFrame->extended_data[1], inFrame1->extended_data[0], inFrame->nb_samples * av_get_bytes_per_sample(AV_SAMPLE_FMT_FLTP));
	*/
	int ret = StoreAudioFrame(outFrame);
	av_frame_free(&outFrame);
	return ret;
}

int StreamFfmpeg::StoreAudioFrame(void* audioFrame)
{
	if (!mInitialized) return 1;
	if (!mHaveAudio) return 1;

	streamThread.lock();

	AVFrame* input_frame = (AVFrame*)audioFrame;
	//mEncodeAudio = !write_audio_frame(formatContext_, &mOutputAudioStream);

	/** Temporary storage for the converted input samples. */
	uint8_t **converted_input_samples = NULL;
	int ret = AVERROR_EXIT;

	/** Initialize the temporary storage for the converted input samples. */
	if (init_converted_samples(&converted_input_samples, mOutputAudioStream.st->codec,
		input_frame->nb_samples))
		goto cleanup;

	/**
	* Convert the input samples to the desired output sample format.
	* This requires a temporary storage provided by converted_input_samples.
	*/
	if (convert_samples((const uint8_t**)input_frame->extended_data, converted_input_samples,
		input_frame->nb_samples, resample_context))
		goto cleanup;

	/** Add the converted input samples to the FIFO buffer for later processing. */
	if (add_samples_to_fifo(fifo, converted_input_samples,
		input_frame->nb_samples))
		goto cleanup;
	ret = 0;

cleanup:
	if (converted_input_samples) {
		av_freep(&converted_input_samples[0]);
		free(converted_input_samples);
	}

	streamThread.unlock();

	return ret;
}

int StreamFfmpeg::StreamAudio()
{
	int error = 0;
	streamThread.lock();
	if (mHaveAudio)
	{
		/** Use the encoder's desired frame size for processing. */
		const int output_frame_size = mOutputAudioStream.st->codec->frame_size;

		/*if (av_compare_ts(mOutputVideoStream.next_pts, mOutputVideoStream.st->codec->time_base,
		mOutputAudioStream.next_pts, mOutputAudioStream.st->codec->time_base) <= 0)
		return 0;*/


		/**
		* If we have enough samples for the encoder, we encode them.
		* At the end of the file, we pass the remaining samples to
		* the encoder.
		*/
		//while (av_audio_fifo_size(fifo) >= output_frame_size)
		if (av_audio_fifo_size(fifo) >= output_frame_size)
			//||	(finished && av_audio_fifo_size(fifo) > 0))
			/**
			* Take one frame worth of audio samples from the FIFO buffer,
			* encode it and write it to the output file.
			*/
		{
			error = load_encode_and_write();
		}
	}
	
	streamThread.unlock();

	return error;
}

int StreamFfmpeg::StreamRemainingSamples()
{
	if (!mHaveAudio) return 0;

	int error = 0;

	int data_written;
	/** Flush the encoder as it may have delayed frames. */
	do {
		error = encode_audio_frame(NULL, &data_written);
		if (error)
			return error;
	} while (data_written);

	return 0;
}

#define FFMIN(a,b) ((a) > (b) ? (b) : (a))

/**
* Load one audio frame from the FIFO buffer, encode and write it to the
* output file.
*/
int StreamFfmpeg::load_encode_and_write()
{
	/** Temporary storage of the output samples of the frame written to the file. */
	AVFrame *output_frame;
	AVCodecContext *output_codec_context = mOutputAudioStream.st->codec;
	/**
	* Use the maximum number of possible samples per frame.
	* If there is less than the maximum possible frame size in the FIFO
	* buffer use this number. Otherwise, use the maximum possible frame size
	*/
	const int frame_size = FFMIN(av_audio_fifo_size(fifo),
		output_codec_context->frame_size);
	int data_written;

	/** Initialize temporary storage for one output frame. */
	if (init_output_frame(&output_frame, output_codec_context, frame_size))
		return AVERROR_EXIT;

	/**
	* Read as many samples from the FIFO buffer as required to fill the frame.
	* The samples are stored in the frame temporarily.
	*/
	if (av_audio_fifo_read(fifo, (void **)output_frame->data, frame_size) < frame_size) {
		fprintf(stderr, "Could not read data from FIFO\n");
		av_frame_free(&output_frame);
		return AVERROR_EXIT;
	}

	/** Encode one frame worth of audio samples. */
	if (encode_audio_frame(output_frame, &data_written)) {
		av_frame_free(&output_frame);
		return AVERROR_EXIT;
	}
	av_frame_free(&output_frame);
	return 0;
}

/** Encode one frame worth of audio to the output file. */
int StreamFfmpeg::encode_audio_frame(AVFrame *frame,
	int *data_present)
{
	/** Packet used for temporary storage. */
	AVPacket output_packet;
	int error;
	init_packet(&output_packet);
	AVCodecContext *output_codec_context = mOutputAudioStream.st->codec;

	/** Set a timestamp based on the sample rate for the container. */
	if (frame) {
		frame->pts = mOutputAudioStream.next_pts;
		mOutputAudioStream.next_pts += frame->nb_samples;
	}

	/**
	* Encode the audio frame and store it in the temporary packet.
	* The output audio stream encoder is used to do this.
	*/
	if ((error = avcodec_encode_audio2(output_codec_context, &output_packet,
		frame, data_present)) < 0) {
		fprintf(stderr, "Could not encode frame (error '%s')\n",
			get_error_text(error));
		av_packet_unref(&output_packet);
		return error;
	}

	/** Write one audio frame from the temporary packet to the output file. */
 	if (*data_present) {
		output_packet.stream_index = mOutputAudioStream.st->id;
		av_packet_rescale_ts(&output_packet, output_codec_context->time_base, mOutputAudioStream.st->time_base);

		//std::cout << "A" << output_packet.pts << " " << output_packet.dts << std::endl;
		//log_packet(formatContext_, &output_packet);
		//av_log_set_level(AV_LOG_DEBUG);
		//if (m_capType == D360::Capture::CaptureDomain::CAPTURE_DSHOW)
		{
			streamMutex.lock();
			if (frame)
			{
				int64_t tempTime = av_rescale_q(frame->pts, output_codec_context->time_base, mOutputVideoStream.st->codec->time_base);
				//std::cout << "\tAudio: " << tempTime << std::endl;
				//if (tempTime > mOutputVideoStream.now_time)
				{
					if (mOutputVideoStream.now_time == -1)
					{
						mOutputVideoStream.begin_offset = output_packet.pts;
					}
					mOutputVideoStream.now_time = tempTime;
					mOutputVideoStream.audio_synced = true;
				}
			}
			streamMutex.unlock();
		}
		output_packet.pts -= mOutputVideoStream.begin_offset;
		output_packet.dts -= mOutputVideoStream.begin_offset;

		output_packet.pts -= audioLag * mOutputAudioStream.st->time_base.den / 1000;
		output_packet.dts -= audioLag * mOutputAudioStream.st->time_base.den / 1000;

		try
		{
			error = av_interleaved_write_frame(formatContext_, &output_packet);
		}
		catch (...)
		{
			error = -1;
		}
		if (error < 0)
		//if ((error = av_interleaved_write_frame(formatContext_, &output_packet)) < 0)
		{
			//fprintf(stderr, "Could not write audio frame (error '%s')\n",
			//	get_error_text(error));
			printf("e");
			av_packet_unref(&output_packet);
			//av_log_set_level(AV_LOG_WARNING);
			return error;
		}
		//av_log_set_level(AV_LOG_WARNING);
		//mOutputAudioStream.st->pts = frame->pts;

		av_packet_unref(&output_packet);
	}

	return 0;
}


//////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////


int StreamFfmpeg::Initialize( std::string inFileName, std::string outFileName )
{
	//av_register_all();

	//
	// Network
	//

	avformat_network_init();

	
	if (inFileName != "")
	{
		initializeContextFromFile(inFileName);
	}

	//
	// Input AVFormatContext and Output AVFormatContext
	//
	const char *out_filename = outFileName.c_str();

	//
	//
	avformat_alloc_output_context2( &formatContext_, NULL, "flv", out_filename ); //RTMP

	//avformat_alloc_output_context2(&formatContext_, NULL, "mpegts", out_filename);//UDP

	if( !formatContext_ )
	{
		std::cout << "Could not create output context" << std::endl;
		errVal = AVERROR_UNKNOWN;
		return error();
	}

	outputFormat_ = formatContext_->oformat;

	if (inFileName != "")
	{
		addStreamsFromInputContext();
	}

	//
	// Dump Format ------------------
	//
	
	av_dump_format( formatContext_, 0, out_filename, 1 );

	//
	// Open output URL
	//
	if( !( outputFormat_->flags & AVFMT_NOFILE ) )
	{
		errVal = avio_open( &formatContext_->pb, out_filename, AVIO_FLAG_WRITE );
		if( errVal < 0 )
		{
			std::cerr << "Could not open output URL " << out_filename << std::endl;
			return error();
		}
	}
	
	time_t rawtime;
	struct tm *timeinfo;
	char buffer[80];

	time( &rawtime );
	timeinfo = localtime( &rawtime );

	strftime( buffer, 80, "%d-%m-%Y %I:%M:%S", timeinfo );

	av_dict_set( &formatContext_->metadata, "creation_time", buffer, 0 );
	av_dict_set( &formatContext_->metadata, "date", buffer, 0 );
	
	errVal = avformat_write_header( formatContext_, 0 );
	if( errVal < 0 )
	{
		return error();
	}

	return 1;
}



int StreamFfmpeg::initializeContextFromFile( std::string fileName )
{
	if( ( errVal = avformat_open_input( &iFormatContext_, fileName.c_str(), 0, 0 ) ) < 0 )
	{
		printf("Could not open input file.");
		return error();
	}
	if(( errVal = avformat_find_stream_info( iFormatContext_, 0 ) ) < 0 )
	{
		printf("Failed to retrieve input stream information");
		return error();
	}
	for( unsigned i = 0; i < iFormatContext_->nb_streams; i++)
	{
		if( iFormatContext_->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			videoIndex = i;
			break;
		}
	}

	av_dump_format( iFormatContext_, 0, fileName.c_str(), 0 );

	return 1;
}


int StreamFfmpeg::addStreamsFromInputContext()
{
	for( unsigned i = 0; i < iFormatContext_->nb_streams; i++ )
	{
		//
		// Create output AVStream according to input AVStream
		//
		AVStream *in_stream = iFormatContext_->streams[i];

		AVCodecContext* c = in_stream->codec;
		if( c->codec_type == AVMEDIA_TYPE_VIDEO )
		{
			videoStream_ = in_stream;
		}
	
		AVStream *out_stream = avformat_new_stream( formatContext_, in_stream->codec->codec );
		if( !out_stream )
		{
			std::cerr << "Failed allocating output stream" << std::endl;;
			errVal = AVERROR_UNKNOWN;
			return error();
		}
		// Copy the settings of AVCodecContext
		errVal = avcodec_copy_context( out_stream->codec, in_stream->codec );
		if( errVal < 0 )
		{
			std::cerr << "Failed to copy context from input to output stream codec context" << std::endl;
			return error();
		}
		out_stream->codec->codec_tag = 0;
		if( formatContext_->oformat->flags & AVFMT_GLOBALHEADER )
			out_stream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
	}

	return 1;
}


int StreamFfmpeg::WriteFrame()
{
	AVPacket pkt;

	//
	// Get an AVPacket
	//
	errVal = av_read_frame( iFormatContext_, &pkt );
	if( errVal < 0 )
		return errVal;


	int64_t start_time = 0;
	//AVCodecContext *c = videoStream_->codec;

	start_time = av_gettime();

	AVStream *in_stream, *out_stream;

	//
	// No PTS (Example: Raw H.264)
	// Simple Write PTS
	//
	if( pkt.pts == AV_NOPTS_VALUE )
	{
		//
		// Write PTS
		//
		AVRational time_base1 = iFormatContext_->streams[ videoIndex ]->time_base;
		//
		// Duration between 2 frames (us)
		//
		int64_t calc_duration = (int64_t)( (double) AV_TIME_BASE / av_q2d( iFormatContext_->streams[ videoIndex ]->r_frame_rate ));
		//
		// Parameters
		//
		pkt.pts = (int64_t)( (double)(frameIndex*calc_duration) / (double)(av_q2d(time_base1)*AV_TIME_BASE) );
		pkt.dts = pkt.pts;
		pkt.duration = (int ) ( ( double )calc_duration / (double)( av_q2d( time_base1 ) * AV_TIME_BASE ) );
	}
	//
	// Important:Delay
	//
	if( pkt.stream_index == videoIndex )
	{
		AVRational time_base = iFormatContext_->streams[ videoIndex ]->time_base;
		AVRational time_base_q = { 1, AV_TIME_BASE };
		int64_t pts_time = av_rescale_q( pkt.dts, time_base, time_base_q );
		int64_t now_time = av_gettime() - start_time;
		if( pts_time > now_time )
			av_usleep( (unsigned int ) ( pts_time - now_time ) );

	}

	in_stream  = iFormatContext_->streams[ pkt.stream_index ];
	out_stream = formatContext_->streams[ pkt.stream_index ];

	//
	// copy packet 
	// PTS/DTS Convert PTS/DTS
	//

	pkt.pts = av_rescale_q_rnd( pkt.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
	pkt.dts = av_rescale_q_rnd( pkt.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
	pkt.duration = (int)av_rescale_q( pkt.duration, in_stream->time_base, out_stream->time_base );
	pkt.pos = -1;
	//
	// Print to Screen
	//
	if ( pkt.stream_index == videoIndex )
	{
		std::cout << "Send %8d video frames to output URL " << frameIndex << std::endl;
		frameIndex++;
	}
	//ret = av_write_frame( formatContext_, &pkt );
	log_packet(formatContext_, &pkt);
	errVal = av_interleaved_write_frame( formatContext_, &pkt );

	if( errVal < 0 )
	{
		std::cerr << "Error muxing packet" << std::endl;
		return 0;
	}

	av_free_packet( &pkt );

	return 1;
}

int StreamFfmpeg::Close()
{
	streamThread.lock();

	StreamRemainingSamples();
	if (formatContext_)
		av_write_trailer( formatContext_ );

	//
	// Close each codec.
	//
	if( mHaveVideo )
		close_stream( formatContext_, &mOutputVideoStream );
	if( mHaveAudio )
		close_stream( formatContext_, &mOutputAudioStream );

	// Close the output file. 
	if ( formatContext_ && !(outputFormat_->flags & AVFMT_NOFILE ) )
		avio_closep( &formatContext_->pb );

	// free the stream 
	if (formatContext_)
		avformat_free_context(formatContext_);

	Clean();

	streamThread.unlock();

	return 1;
}

int StreamFfmpeg::error()
{
	char errBuf[ AV_ERROR_MAX_STRING_SIZE ];
	if( errVal < 0 && errVal != AVERROR_EOF )
	{
		char* errStr = av_make_error_string( errBuf, AV_ERROR_MAX_STRING_SIZE, errVal );
		std::cerr << errStr << std::endl;
	}

	avformat_close_input( &iFormatContext_ );
	//
	// close output 
	//
	if( formatContext_ && !( outputFormat_->flags & AVFMT_NOFILE ) )
		avio_close( formatContext_->pb );
	avformat_free_context( formatContext_ );

	return errVal;
};

