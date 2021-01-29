
#ifndef __BASE_FFMPEG__
#define __BASE_FFMPEG__

#include <string>
#include <QString>

extern "C" {
#include "libavformat/avformat.h"
#include "libavdevice/avdevice.h"
}


typedef enum AUDIO_CHANNEL_TYPE
{
	AUDIO_CHANNEL_TYPE_NO_AUDIO = 100,
	AUDIO_CHANNEL_TYPE_LEFT,
	AUDIO_CHANNEL_TYPE_RIGHT,
	AUDIO_CHANNEL_TYPE_MIXED,
	
	// Error
	AUDIO_CHANNEL_TYPE_ERROR_NO_FILE,
	AUDIO_CHANNEL_TYPE_ERROR_FAILED_RETRIEVE

} AUDIO_CHANNEL_TYPE;

class BaseFfmpeg
{
public:
	BaseFfmpeg();	
	virtual ~BaseFfmpeg(void);

	AUDIO_CHANNEL_TYPE checkAudioChannelInVideoFile(QString videoFilePath);

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
};

#endif // __BASE_FFMPEG__