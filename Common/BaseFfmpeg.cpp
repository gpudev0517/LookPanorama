#include "BaseFfmpeg.hpp"

BaseFfmpeg::BaseFfmpeg()
{
	avcodec_register_all();
	av_register_all();
	//av_log_set_level(AV_LOG_QUIET);

	totalFrames_ = 0;

	outputFormat_ = 0;
	formatContext_ = 0;
	codecContext_ = 0;
	videoStream_ = 0;
}

BaseFfmpeg::~BaseFfmpeg()
{

}

AUDIO_CHANNEL_TYPE BaseFfmpeg::checkAudioChannelInVideoFile(QString videoFilePath)
{
	avdevice_register_all();

	//videoFilePath = "d:/sport17-test.mp4";
	//videoFilePath = "d:/test2.mp4";

	AVFormatContext* m_pFormatCtx = avformat_alloc_context();
	AVInputFormat *fmt = NULL;

	AVDictionary        *tOptions = NULL;
	/*char fpsString[10]; int fps = 25;
	sprintf(fpsString, "%d", (int)fps);
	av_dict_set(&tOptions, "framerate", fpsString, 0);*/
	av_dict_set(&tOptions, "rtbufsize", "100000000", 0);

	if (avformat_open_input(&m_pFormatCtx, videoFilePath.toUtf8().data(), fmt, &tOptions) < 0)
	{
		fprintf(stderr, "Could not open input file '%s'", videoFilePath.toUtf8().data());
		return AUDIO_CHANNEL_TYPE_ERROR_NO_FILE;
	}
	if ((avformat_find_stream_info(m_pFormatCtx, 0)) < 0)
	{
		fprintf(stderr, "Failed to retrieve input stream information");
		return AUDIO_CHANNEL_TYPE_ERROR_FAILED_RETRIEVE;
	}

	AUDIO_CHANNEL_TYPE result = AUDIO_CHANNEL_TYPE_NO_AUDIO;
	int audioStream = -1;
	for (int i = 0; i < m_pFormatCtx->nb_streams; i++) {
		// check if audio channel exists in video file
		if (m_pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO &&
			audioStream < 0) {
			audioStream = i;
		}

		// check audio's channel layout
		if (audioStream >= 0) {
			int audio_channel_layout = m_pFormatCtx->streams[i]->codec->channel_layout;
			if (audio_channel_layout != 0) {
				if (audio_channel_layout == AV_CH_FRONT_LEFT)
					result = AUDIO_CHANNEL_TYPE_LEFT;
				else if (audio_channel_layout == AV_CH_FRONT_RIGHT)
					result = AUDIO_CHANNEL_TYPE_RIGHT;				
				else
					result = AUDIO_CHANNEL_TYPE_LEFT;		// Temporarily return LEFT if audio type is stereo, cause by none implemented left and right channel mixing feature.
					//result = AUDIO_CHANNEL_TYPE_MIXED;
			} else {				
				result = AUDIO_CHANNEL_TYPE_MIXED;
			}
		} 
		else
		{
			result = AUDIO_CHANNEL_TYPE_NO_AUDIO;
		}
	}
	if (audioStream == -1)
	{
		fprintf(stderr, "RAH couldn't find audio stream");
		result = AUDIO_CHANNEL_TYPE_NO_AUDIO;
	}

	return result;
}