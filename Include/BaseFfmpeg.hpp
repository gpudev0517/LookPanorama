
#ifndef __BASE_FFMPEG__
#define __BASE_FFMPEG__

#include <string>

extern "C" {
#include "libavformat/avformat.h"
}


class BaseFfmpeg
{
protected:
	AVFormatContext *formatContext_;
	AVCodecContext  *codecContext_;
	AVOutputFormat  *outputFormat_;
	AVStream        *videoStream_;
	AVStream		*audioStream_;

	AVFrame         *frame_;
	AVPicture       picture_;

	AVPacket		packet_;

	int totalFrames_;

	BaseFfmpeg() 
	{
		avcodec_register_all();
		av_register_all();
		av_log_set_level(AV_LOG_QUIET);

		totalFrames_  = 0;

		outputFormat_ = 0;
		formatContext_  = 0;
		codecContext_   = 0;
		videoStream_ = 0;
	};

	~BaseFfmpeg(void) { };
};

#endif // __BASE_FFMPEG__