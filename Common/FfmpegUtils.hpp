#ifdef _WIN32
#define snprintf sprintf_s
#endif

extern "C"
{

#include <libavutil/time.h>
#include <libavutil/error.h>
#include <libavutil/avassert.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
};

#define CLIP(X) ( (X) > 255 ? 255 : (X) < 0 ? 0 : X)

// RGB -> YUV
#define RGB2Y(R, G, B) CLIP(( (  66 * (R) + 129 * (G) +  25 * (B) + 128) >> 8) +  16)
#define RGB2U(R, G, B) CLIP(( ( -38 * (R) -  74 * (G) + 112 * (B) + 128) >> 8) + 128)
#define RGB2V(R, G, B) CLIP(( ( 112 * (R) -  94 * (G) -  18 * (B) + 128) >> 8) + 128)

// YUV -> RGB
#define C(Y) ( (Y) - 16  )
#define D(U) ( (U) - 128 )
#define E(V) ( (V) - 128 )

#define YUV2R(Y, U, V) CLIP(( 298 * C(Y)              + 409 * E(V) + 128) >> 8)
#define YUV2G(Y, U, V) CLIP(( 298 * C(Y) - 100 * D(U) - 208 * E(V) + 128) >> 8)
#define YUV2B(Y, U, V) CLIP(( 298 * C(Y) + 516 * D(U)              + 128) >> 8)

// RGB -> YCbCr
#define CRGB2Y(R, G, B) CLIP((19595 * R + 38470 * G + 7471 * B ) >> 16)
#define CRGB2Cb(R, G, B) CLIP((36962 * (B - CLIP((19595 * R + 38470 * G + 7471 * B ) >> 16) ) >> 16) + 128)
#define CRGB2Cr(R, G, B) CLIP((46727 * (R - CLIP((19595 * R + 38470 * G + 7471 * B ) >> 16) ) >> 16) + 128)

// YCbCr -> RGB
#define CYCbCr2R(Y, Cb, Cr) CLIP( Y + ( 91881 * Cr >> 16 ) - 179 )
#define CYCbCr2G(Y, Cb, Cr) CLIP( Y - (( 22544 * Cb + 46793 * Cr ) >> 16) + 135)
#define CYCbCr2B(Y, Cb, Cr) CLIP( Y + (116129 * Cb >> 16 ) - 226 )

#define STREAM_DURATION   200.0
#define STREAM_FRAME_RATE 30 /* 25 images/s */
#define STREAM_NB_FRAMES  ((int)(STREAM_DURATION * STREAM_FRAME_RATE))
#define STREAM_PIX_FMT    AV_PIX_FMT_YUV420P/* default pix_fmt */


#define FILE_PIX_FMT	  AV_PIX_FMT_YUV420P
#define LIDAR_PIX_FMT		AV_PIX_FMT_YUV420P//AV_PIX_FMT_GRAY16LE

#define X265_MAX_FRAME_THREADS  16


static void log_packet(const AVFormatContext *fmt_ctx, const AVPacket *pkt)
{
	AVRational *time_base = &fmt_ctx->streams[pkt->stream_index]->time_base;

	/*printf("pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s stream_index:%d\n",
	av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, time_base),
	av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, time_base),
	av_ts2str(pkt->duration), av_ts2timestr(pkt->duration, time_base),
	pkt->stream_index);*/
	char str1[100];
	av_ts_make_string(str1, pkt->pts);
	char str2[100];
	av_ts_make_time_string(str2, pkt->pts, time_base);
	char str3[100];
	av_ts_make_string(str3, pkt->dts);
	char str4[100];
	av_ts_make_time_string(str4, pkt->dts, time_base);
	char str5[100];
	av_ts_make_string(str5, pkt->duration);
	char str6[100];
	av_ts_make_time_string(str6, pkt->duration, time_base);

	printf("pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s stream_index:%d\n",
		str1, str2,
		str3, str4,
		str5, str6,
		pkt->stream_index);
}


static int write_frame( AVFormatContext *fmt_ctx, const AVRational *time_base, AVStream *st, AVPacket *pkt )
{
	// rescale output packet timestamp values from codec to stream timebase 
	av_packet_rescale_ts( pkt, *time_base, st->time_base );
	//std::cout << "V" << pkt->pts << "\t" << pkt->dts << std::endl;
	pkt->stream_index = st->index;
	//log_packet(fmt_ctx, pkt);
	//return true;
	return av_interleaved_write_frame(fmt_ctx, pkt);
}


static int add_stream(OutputStream *ost, AVFormatContext *oc, AVCodec **codec, enum AVCodecID codec_id,
	int width, int height, int video_fps, bool toFile, int audioSampleRate, int channels, bool bLidar)
{
	AVCodecContext *c;
	int i;

	/* find the encoder */
	*codec = avcodec_find_encoder( codec_id );
	if (!(*codec)) {
		std::cerr << "Could not find encoder for " <<  avcodec_get_name( codec_id );
		return -1;
	}

	ost->st = avformat_new_stream( oc, *codec );
	if( !ost->st ) 
	{
		std::cerr << "Could not allocate stream" << std::endl;;
		return -1;
	}
	ost->st->id = oc->nb_streams - 1;
	c = ost->st->codec;

	AVRational r;
	switch( (*codec)->type ) 
	{
	case AVMEDIA_TYPE_AUDIO:
		c->sample_fmt = (*codec)->sample_fmts ? (*codec)->sample_fmts[0] : AV_SAMPLE_FMT_FLTP; // AV_SAMPLE_FMT_FLTP for video file, AV_SAMPLE_FMT_S16 for microphone
		c->bit_rate    = 96000;
		c->sample_rate = audioSampleRate; //48000 for video file, 44100 for microphone
		/*if ((*codec)->supported_samplerates) 
		{
			c->sample_rate = (*codec)->supported_samplerates[0];
			for (i = 0; (*codec)->supported_samplerates[i]; i++) 
			{
				if ((*codec)->supported_samplerates[i] == 44100)
					c->sample_rate = 44100;
			}
		}*/
		/*
		c->channels = av_get_channel_layout_nb_channels(c->channel_layout);
		c->channel_layout = AV_CH_LAYOUT_STEREO;
		if ((*codec)->channel_layouts) 
		{
			c->channel_layout = (*codec)->channel_layouts[0];
			for (i = 0; (*codec)->channel_layouts[i]; i++) 
			{
				if ((*codec)->channel_layouts[i] == AV_CH_LAYOUT_STEREO)
					c->channel_layout = AV_CH_LAYOUT_STEREO;
			}
		}
		c->channels = av_get_channel_layout_nb_channels(c->channel_layout);
		*/
		c->channels = channels;
		c->channel_layout = av_get_default_channel_layout(channels);
	
		r.num = 1;
		r.den = c->sample_rate;
		ost->st->time_base = r;

		/** Allow the use of the experimental AAC encoder */
		c->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;
		break;

	case AVMEDIA_TYPE_VIDEO:
		c->codec_id = codec_id;

		//c->bit_rate = width * height / 1.0f;
		c->bit_rate = 1500 * 1000 * 4; // 1500kbps
		//c->bit_rate	= 400000;
		// Resolution must be a multiple of two. 
		c->width	= width;
		c->height	= height;
		//////////////////////////////////////////////////////////////////////////
		// timebase: This is the fundamental unit of time (in seconds) in terms
		// of which frame timestamps are represented. For fixed-fps content,
		// timebase should be 1/framerate and timestamp increments should be
		// identical to 1. 
		//
#if 0
		r.num				= 1;
		r.den				= STREAM_FRAME_RATE;
		ost->st->time_base	= r;
		c->time_base		= ost->st->time_base;

		c->gop_size	= 12; 
		c->pix_fmt	= STREAM_PIX_FMT;
		if( c->codec_id == AV_CODEC_ID_MPEG2VIDEO ) 
		{
			c->max_b_frames = 2;
		}
		if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO) 
		{
			c->mb_decision = 2;
		}

#else
		c->codec_type = AVMEDIA_TYPE_VIDEO;
		//c->profile = FF_PROFILE_H264_MAIN;
		//c->bit_rate = 0;
		c->coded_width = c->width = width;
		c->coded_height = c->height = height;

		c->sample_aspect_ratio.den = 1;
		c->sample_aspect_ratio.num = 1;
		c->time_base.den = video_fps;
		if(bLidar)
			c->time_base.den = 5;
		c->time_base.num = 1;
		ost->st->time_base = c->time_base;
		//c->ticks_per_frame = 2;
		c->pkt_timebase.den = 1000;
		c->pkt_timebase.num = 1;
		if( toFile )
			c->gop_size = 75;
		else
			c->gop_size = 12;
		c->max_b_frames = 2;

		if( codec_id == AV_CODEC_ID_H265 )
			c->thread_count = X265_MAX_FRAME_THREADS;
		else
			c->thread_count = 18;

		c->keyint_min = 38;
		
		if( bLidar )
			c->pix_fmt = LIDAR_PIX_FMT;
		else
			c->pix_fmt = FILE_PIX_FMT;
		
		//c->profile = 100;

		
		/*//c->refs = 0;
		c->me_method = ME_EPZS; //1 (zero), 2 (full), 3 (log), 4 (phods), 5 (epzs), 6 (x1), 7 (hex), 8 (umh), 9 (iter), 10 (tesa)[7, 8, 10 are x264 specific, 9 is snow specific]
		c->trellis = 0;
		//c->chromaoffset = 0;
		
		//c->me_sub_cmp = 2;

		av_opt_set(c, "subme", "2", 0);
		av_opt_set(c, "mixed_ref", "1", 0);
		av_opt_set(c, "chroma_qp_offset", "0", 0);
		av_opt_set(c, "lookahead_threads", "3", 0);*/
#endif

		break;

	default:
		break;
	}


	if( oc->oformat->flags & AVFMT_GLOBALHEADER )
		c->flags |= CODEC_FLAG_GLOBAL_HEADER;


	return 1;
}




static AVFrame *alloc_audio_frame( enum AVSampleFormat sample_fmt, uint64_t channel_layout, int sample_rate, int nb_samples )
{
	AVFrame *frame = av_frame_alloc();
	int ret;

	if( !frame ) 
	{
		std::cerr << "Error allocating an audio frame" << std::endl;
		return NULL;
	}

	frame->format = sample_fmt;
	frame->channel_layout = channel_layout;
	frame->sample_rate = sample_rate;
	frame->nb_samples = nb_samples;

	if( nb_samples ) 
	{
		ret = av_frame_get_buffer( frame, 0 );
		if( ret < 0 ) 
		{
			std::cerr << "Error allocating an audio buffer " << std::endl;
			return NULL;
		}
	}

	return frame;
}

//////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////


static int open_audio( AVFormatContext *oc, AVCodec *codec, OutputStream *ost, AVDictionary *opt_arg )
{
	AVCodecContext *c;
	int nb_samples;
	int ret;
	AVDictionary *opt = NULL;

	if (ost->st)
		c = ost->st->codec;
	else
		return -1;

	/// open it 
	av_dict_copy( &opt, opt_arg, 0 );
	ret = avcodec_open2( c, codec, &opt );
	av_dict_free( &opt );
	if( ret < 0 ) 
	{
		std::cerr << "Could not open audio codec:" << std::endl;
		return ret;
	}

	return 0;

	// init signal generator 

	ost->t = 0;
	ost->tincr = (float)(2 * M_PI * 110.0 / c->sample_rate);
	// increment frequency by 110 Hz per second 
	ost->tincr2 = (float)(2 * M_PI * 110.0 / c->sample_rate / c->sample_rate);

	if( c->codec->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE )
		nb_samples = 10000;
	else
		nb_samples = c->frame_size;

	ost->frame = alloc_audio_frame( c->sample_fmt, c->channel_layout, c->sample_rate, nb_samples );
	ost->tmp_frame = alloc_audio_frame( AV_SAMPLE_FMT_S16, c->channel_layout, c->sample_rate, nb_samples );

	//
	// create re-sampler context 
	//
	ost->swr_ctx = swr_alloc();
	if( !ost->swr_ctx ) 
	{
		std::cerr << "Could not allocate re-sampler context" << std::endl;
		return -1;
	}

	//
	// set options 
	//
	av_opt_set_int( ost->swr_ctx, "in_channel_count", c->channels, 0 );
	av_opt_set_int( ost->swr_ctx, "in_sample_rate", c->sample_rate, 0 );
	av_opt_set_sample_fmt( ost->swr_ctx, "in_sample_fmt", AV_SAMPLE_FMT_S16, 0 );
	av_opt_set_int( ost->swr_ctx, "out_channel_count", c->channels, 0 );
	av_opt_set_int( ost->swr_ctx, "out_sample_rate", c->sample_rate, 0 );
	av_opt_set_sample_fmt( ost->swr_ctx, "out_sample_fmt", c->sample_fmt, 0 );

	//
	// initialize the re-sampling context 
	//
	if(( ret = swr_init(ost->swr_ctx)) < 0 ) 
	{
		std::cerr << "Failed to initialize the re-sampling context" << std::endl;
		return -1;
	}
}

//////////////////////////////////////////////////////////////////////////
// Prepare a 16 bit dummy audio frame of 'frame_size' samples and
// 'nb_channels' channels. 
//////////////////////////////////////////////////////////////////////////

static AVFrame *get_audio_frame(OutputStream *ost)
{
	AVFrame *frame = ost->tmp_frame;
	int j, i, v;
	int16_t *q = 0;
	
	if (frame)
		q = (int16_t*)frame->data[0];
	else
		return NULL;

	// check if we want to generate more frames 

	AVRational r;
	r.num = 1;
	r.den = 1;
	if( av_compare_ts( (int64_t) ost->next_pts, ost->st->codec->time_base, (int64_t) STREAM_DURATION, r) >= 0 )
		return NULL;

	for( j = 0; j < frame->nb_samples; j++ ) 
	{
		v = (int) (sin( ost->t ) * 10000 );
		for( i = 0; i < ost->st->codec->channels; i++ )
			*q++ = v;
		ost->t     += ost->tincr;
		ost->tincr += ost->tincr2;
	}

	frame->pts     = ost->next_pts;
	ost->next_pts += frame->nb_samples;

	return frame;
}

//
// encode one audio frame and send it to the muxer
// return 1 when encoding is finished, 0 otherwise
//

static int write_audio_frame(AVFormatContext *oc, OutputStream *ost)
{
	AVCodecContext *c;
	AVPacket pkt = { 0 }; // data and size must be 0;
	AVFrame *frame;
	int ret;
	int got_packet;
	int dst_nb_samples;

	av_init_packet( &pkt );
	c = ost->st->codec;

	frame = get_audio_frame( ost );

	if( frame ) 
	{
		// convert samples from native format to destination codec format, using the re-sampler 
		// compute destination number of samples 
		dst_nb_samples = av_rescale_rnd((int64_t) (swr_get_delay(ost->swr_ctx, (int64_t)c->sample_rate) + (int64_t) frame->nb_samples),
			(int64_t)c->sample_rate, (int64_t)c->sample_rate, (AVRounding) AV_ROUND_UP);
		av_assert0( dst_nb_samples == frame->nb_samples );

		// when we pass a frame to the encoder, it may keep a reference to it
		// internally;
		// make sure we do not overwrite it here
		//
		ret = av_frame_make_writable( ost->frame );
		if( ret < 0 )
			return ret;

		//
		// convert to destination format 
		//
		ret = swr_convert( ost->swr_ctx, ost->frame->data, dst_nb_samples, (const uint8_t **)frame->data, frame->nb_samples );
		if( ret < 0 ) 
		{
			std::cerr << "Error while converting" << std::endl;
			return ret;
		}
		frame = ost->frame;

		AVRational r;
		r.num = 1;
		r.den = c->sample_rate;
		frame->pts = av_rescale_q( ost->samples_count, r, c->time_base );
		ost->samples_count += dst_nb_samples;
	}

	ret = avcodec_encode_audio2( c, &pkt, frame, &got_packet );
	if (ret < 0) 
	{
		std::cerr << "Error encoding audio frame: " << std::endl;
		return ret;
	}

	if( got_packet ) 
	{
		ret = write_frame(oc, &c->time_base, ost->st, &pkt);
		if (ret < 0) 
		{
			std::cerr << "e";
			return 0;
		}
	}

	return (frame || got_packet) ? 0 : 1;
}

//////////////////////////////////////////////////////////////////////////
// video output 
//////////////////////////////////////////////////////////////////////////

static AVFrame *alloc_picture( enum AVPixelFormat pix_fmt, int width, int height )
{
	AVFrame *picture;
	int ret;

	picture = av_frame_alloc();
	if (!picture)
		return NULL;

	picture->format = pix_fmt;
	picture->width = width;
	picture->height = height;

	//
	// allocate the buffers for the frame data 
	//
	ret = av_frame_get_buffer( picture, 32 );
	if (ret < 0) 
	{
		std::cerr << "Could not allocate frame data " << std::endl;
		return NULL;
	}

	return picture;
}

static int open_video( AVFormatContext *oc, AVCodec *codec, OutputStream *ost, AVDictionary *opt_arg )
{
	int ret;
	AVCodecContext *c = NULL; 
	AVDictionary *opt = NULL;

	av_dict_copy( &opt, opt_arg, 0 );

	if (ost->st)
		c = ost->st->codec;
	else
		return -1;

	//av_log_set_level(AV_LOG_ERROR);
	//
	// open the codec 
	//
	ret = avcodec_open2( c, codec, &opt );
	av_dict_free( &opt );
	if( ret < 0 ) 
	{
		std::cerr << "Could not open video codec: " << std::endl;
		return ret;
	}

	//
	// allocate and init a re-usable frame 
	//
	ost->frame = alloc_picture( c->pix_fmt, c->width, c->height );
	if( !ost->frame ) 
	{
		std::cerr << "Could not allocate video frame " << std::endl;
		return -1;
	}

	//
	// If the output format is not YUV420P, then a temporary YUV420P
	// picture is needed too. It is then converted to the required
	// output format.
	//
	ost->tmp_frame = NULL;
	if( c->pix_fmt != AV_PIX_FMT_YUV420P ) 
	{
		ost->tmp_frame = alloc_picture( AV_PIX_FMT_YUV420P, c->width, c->height );
		if (!ost->tmp_frame) 
		{
			std::cerr << "Could not allocate temporary picture" << std::endl;
			return -1;
		}
	}
	return 0;
}

//
// Prepare a dummy image. 
//
static void fill_yuv_image( AVFrame *pict, int frame_index, int width, int height )
{
	int x, y, i, ret;

	//
	// when we pass a frame to the encoder, it may keep a reference to it
	// internally;
	// make sure we do not overwrite it here
	//
	ret = av_frame_make_writable( pict );
	if( ret < 0 )
		return;

	i = frame_index;

	// Y 
	for( y = 0; y < height; y++ )
	{
		for( x = 0; x < width; x++ )
		{
			pict->data[0][y * pict->linesize[0] + x] = x + y + i * 3;
		}
	}

	// Cb and Cr 
	for( y = 0; y < height / 2; y++ ) 
	{
		for( x = 0; x < width / 2; x++ ) 
		{
			pict->data[1][y * pict->linesize[1] + x] = 128 + y + i * 2;
			pict->data[2][y * pict->linesize[2] + x] = 64 + x + i * 5;
		}
	}
}

static void flip_frame(AVFrame* pFrame) 
{
	for( int i = 0; i < 1; i++) 
	{
		pFrame->data[i]			+= pFrame->linesize[i] * (pFrame->height - 1);
		pFrame->linesize[i]		= -pFrame->linesize[i];
	}
}

static void unflip_frame( AVFrame* pFrame ) 
{
	for( int i = 0; i < 1; i++ ) 
	{
		pFrame->linesize[i]		= -pFrame->linesize[i];
		pFrame->data[i]			-= pFrame->linesize[i] * (pFrame->height - 1);
	}
}

#define SCALE_FLAGS SWS_BICUBIC

static AVFrame *get_video_frame(OutputStream *ost, uint8_t* data, bool isAudioSync)
{
	AVCodecContext *c = ost->st->codec;

	//
	// check if we want to generate more frames 
	//
	//AVRational r;
	//r.num = 1;
	//r.den = 1; 
	//if (av_compare_ts((int64_t)ost->next_pts, ost->st->codec->time_base, (int64_t)STREAM_DURATION, r) >= 0)
	//	return NULL;

	/*if( c->pix_fmt != AV_PIX_FMT_YUV420P ) 
	{
		//
		// as we only generate a YUV420P picture, we must convert it
		// to the codec pixel format if needed 
		//
		if( !ost->sws_ctx ) 
		{
			ost->sws_ctx = sws_getContext( c->width, c->height, AV_PIX_FMT_YUV420P, c->width, c->height, c->pix_fmt,
											SCALE_FLAGS, NULL, NULL, NULL );
			if( !ost->sws_ctx ) 
			{
				std::cerr << "Could not initialize the conversion context" << std::endl;
				return NULL;
			}
		}
		fill_yuv_image( ost->tmp_frame, ost->next_pts, c->width, c->height );
		sws_scale( ost->sws_ctx, (const uint8_t * const *)ost->tmp_frame->data, 
					ost->tmp_frame->linesize, 0, c->height, 
					ost->frame->data, ost->frame->linesize );
	}
	else */
	{
		if (data)
		{
#if 1
			//SwsContext * ctx = sws_getContext(c->width, c->height, AV_PIX_FMT_BGRA, c->width, c->height, AV_PIX_FMT_YUV420P, 0, 0, 0, 0);
			if (c->pix_fmt == AV_PIX_FMT_YUV420P)
			{
				int output_widths[3] = { c->width, c->width / 2, c->width / 2 };
				int output_heights[3] = { c->height, c->height / 2, c->height / 2 };
				int output_offsets[3] = { 0, c->width * c->height, c->width * c->height + c->width / 2 };

				for (int k = 0; k < 3; k++)
				{
					if (k == 0 && output_widths[k] == ost->frame->linesize[k])
					{
						memcpy(ost->frame->data[0], data + output_offsets[0], output_widths[0] * output_heights[0]);
					}
					else
					{
						for (int i = 0; i < output_heights[k]; i++)
						{
							memcpy(ost->frame->data[k] + i * ost->frame->linesize[k], data + output_offsets[k] + i * c->width, output_widths[k]);
						}
					}
				}
			}
			else if (c->pix_fmt == AV_PIX_FMT_YUV444P)
			{
				ost->frame->data[0] = data;
				ost->frame->data[1] = data + c->width * c->height;
				ost->frame->data[2] = data + c->width * c->height * 2;
			}
			
			//flip_frame( ost->frame );
#else
			uint8_t * ydata = ost->frame->data[0];
			uint8_t * udata = ost->frame->data[1];
			uint8_t * vdata = ost->frame->data[2];
			uint8_t * srcbuf = data;
			uint8_t * ybuf = ydata;

			for (int i = 0; i < c->height; i++)
			{
				for (int j = 0; j < c->width; j++)
				{
					uint8_t r = *(srcbuf++);
					uint8_t g = *(srcbuf++);
					uint8_t b = *(srcbuf++);

					*(ybuf++) = r;
					udata[(i / 2) * c->width / 2 + j / 2] = g;
					vdata[(i / 2) * c->width / 2 + j / 2] = b;

					//uint8_t y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
					//uint8_t u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
					//uint8_t v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

					//ybuf[i * c->width + j] = y;
					//ubuf[(i / 2) * c->width / 2 + j / 2] = u;
					//vbuf[(i / 2) * c->width / 2 + j / 2] = v;
				}
			}
			//memcpy(ybuf, data, c->height * c->width);
			//memcpy(ubuf, data, c->height * c->width / 4);
			//memcpy(vbuf, data, c->height * c->width / 4);
#endif
		}
		else
		{
			fill_yuv_image( ost->frame, (int)ost->next_pts, (int)c->width, (int)c->height );
		}
	}

	ost->frame->pts = ost->next_pts++;
	//std::cout << "\tVideo: " << ost->frame->pts << std::endl;
	//if (m_sharedImageBuffer->getGlobalAnimSettings()->m_captureType == D360::Capture::CaptureDomain::CAPTURE_DSHOW)
	if (isAudioSync)
	{
		int64_t time_diff = ost->next_pts - ost->now_time;
		if (time_diff > 100 || time_diff < -100)
		{
			std::cout << "S";
			ost->next_pts = ost->now_time;
		}
	}

	return ost->frame;
}

//
// encode one video frame and send it to the muxer
// return 1 when encoding is finished, 0 otherwise
//
static int write_video_frame( AVFormatContext *oc, OutputStream *ost, uint8_t* data, int frameIndex, int fps, bool isAudioSync )
{
	int ret;
	AVCodecContext *c;
	AVFrame *frame;
	int got_packet = 0;

	if (ost->st)
		c = ost->st->codec;
	else
		return -1;
	//av_log_set_level(AV_LOG_INFO);
	frame = get_video_frame( ost, data, isAudioSync );

	if( oc->oformat->flags & AVFMT_RAWPICTURE ) 
	{
		// a hack to avoid data copy with some raw video muxers 
		AVPacket pkt;
		av_init_packet( &pkt );

		if( !frame )
			return 1;

		pkt.flags		|= AV_PKT_FLAG_KEY;
		pkt.stream_index = ost->st->index;
		pkt.data		= (uint8_t *)frame;
		pkt.size		= sizeof( AVPicture );

		pkt.pts = pkt.dts = frame->pts;
		av_packet_rescale_ts( &pkt, c->time_base, ost->st->time_base );

		ret = av_interleaved_write_frame( oc, &pkt );
	}
	else 
	{
		AVPacket pkt = { 0 };
		av_init_packet( &pkt );

		/* encode the image */
		ret = avcodec_encode_video2( c, &pkt, frame, &got_packet );
		if( ret < 0 ) 
		{
			std::cerr << "Error encoding video frame: " << std::endl;
			return ret;
		}

		//
		// No PTS (Example: Raw H.264)
		// Simple Write PTS
		//
		if (pkt.pts == AV_NOPTS_VALUE)
		{
			/*//
			// Write PTS
			//
			AVRational time_base1 = ost->st->time_base;
			//
			// Duration between 2 frames (us)
			//
			int64_t calc_duration = (int64_t)((double)AV_TIME_BASE / fps);
			//
			// Parameters
			//
			pkt.pts = (int64_t)((double)(frameIndex*calc_duration) / (double)(av_q2d(time_base1)*AV_TIME_BASE));
			pkt.dts = pkt.pts;
			pkt.duration = (int)((double)calc_duration / (double)(av_q2d(time_base1) * AV_TIME_BASE));*/
			pkt.pts = 0;
			pkt.dts = 0;
			pkt.duration = 0;
		}
		else
		{
			/*if (ost->now_time > -1) {
				if (ost->audio_synced)
				{
					ost->audio_synced = false;
				}
				else
				{
					ost->now_time++;
				}
				pkt.pts = ost->now_time;
				pkt.dts = pkt.pts;
			}*/
		}

#if 0
		int key_frame = 1;
		pkt.pts = (frameIndex + 1) * key_frame;
		pkt.dts = frameIndex * key_frame;
		pkt.duration = key_frame;
		pkt.pos = -1;

		//pkt.dts = AV_NOPTS_VALUE;
		//pkt.pts = AV_NOPTS_VALUE;
#endif
		if( got_packet ) 
		{
			ret = write_frame( oc, &c->time_base, ost->st, &pkt );
		}
		else {
			//printf("_");
			ret = 0;
		}
	}

	if( ret < 0 ) 
	{
		std::cerr << "E";
		return 0;
	}

	//av_log_set_level(AV_LOG_WARNING);

	return (frame || got_packet) ? 0 : 1;
}

static AVFrame* WebRTC_write_video_frame(AVFormatContext *oc, OutputStream *ost, uint8_t* data, int frameIndex, int fps, bool isAudioSync)
{
	int ret;
	AVCodecContext *c;
	AVFrame *frame;
	int got_packet = 0;

	if (ost->st)
		c = ost->st->codec;
	else
		return nullptr;

	//av_log_set_level(AV_LOG_INFO);
	frame = get_video_frame(ost, data, isAudioSync);

#if 0
	if (oc->oformat->flags & AVFMT_RAWPICTURE)
	{
		// a hack to avoid data copy with some raw video muxers 
		AVPacket pkt;
		av_init_packet(&pkt);

		if (!frame)
			return 1;

		pkt.flags |= AV_PKT_FLAG_KEY;
		pkt.stream_index = ost->st->index;
		pkt.data = (uint8_t *)frame;
		pkt.size = sizeof(AVPicture);

		pkt.pts = pkt.dts = frame->pts;
		av_packet_rescale_ts(&pkt, c->time_base, ost->st->time_base);

		ret = av_interleaved_write_frame(oc, &pkt);
	}
	else
	{
		AVPacket pkt = { 0 };
		av_init_packet(&pkt);

		/* encode the image */
		ret = avcodec_encode_video2(c, &pkt, frame, &got_packet);
		if (ret < 0)
		{
			std::cerr << "Error encoding video frame: " << std::endl;
			return ret;
		}

		//
		// No PTS (Example: Raw H.264)
		// Simple Write PTS
		//
		if (pkt.pts == AV_NOPTS_VALUE)
		{
			/*//
			// Write PTS
			//
			AVRational time_base1 = ost->st->time_base;
			//
			// Duration between 2 frames (us)
			//
			int64_t calc_duration = (int64_t)((double)AV_TIME_BASE / fps);
			//
			// Parameters
			//
			pkt.pts = (int64_t)((double)(frameIndex*calc_duration) / (double)(av_q2d(time_base1)*AV_TIME_BASE));
			pkt.dts = pkt.pts;
			pkt.duration = (int)((double)calc_duration / (double)(av_q2d(time_base1) * AV_TIME_BASE));*/
			pkt.pts = 0;
			pkt.dts = 0;
			pkt.duration = 0;
		}
		else
		{
			/*if (ost->now_time > -1) {
			if (ost->audio_synced)
			{
			ost->audio_synced = false;
			}
			else
			{
			ost->now_time++;
			}
			pkt.pts = ost->now_time;
			pkt.dts = pkt.pts;
			}*/
		}

#if 0
		int key_frame = 1;
		pkt.pts = (frameIndex + 1) * key_frame;
		pkt.dts = frameIndex * key_frame;
		pkt.duration = key_frame;
		pkt.pos = -1;

		//pkt.dts = AV_NOPTS_VALUE;
		//pkt.pts = AV_NOPTS_VALUE;
#endif
		if (got_packet)
		{
			ret = write_frame(oc, &c->time_base, ost->st, &pkt);
		}
		else {
			//printf("_");
			ret = 0;
		}
	}

	if (ret < 0)
	{
		std::cerr << "E";
		return 0;
	}

	//av_log_set_level(AV_LOG_WARNING);

	return (frame || got_packet) ? 0 : 1;
#endif
	return frame;
}

static void close_stream(AVFormatContext *oc, OutputStream *ost)
{
	if (ost->st && ost->st->codec)
		avcodec_close(ost->st->codec);

	if (ost->frame)
		av_frame_free(&ost->frame);

	if (ost->tmp_frame)
		av_frame_free(&ost->tmp_frame);

	if (ost->sws_ctx)
		sws_freeContext(ost->sws_ctx);
	if (ost->swr_ctx)
		swr_free(&ost->swr_ctx);
}