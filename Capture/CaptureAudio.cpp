#include "CaptureAudio.h"





int AudioInput::getSampleRate()
{
	if (m_pAudioCodecCtx)
		return m_pAudioCodecCtx->sample_rate;
	return 48000;
}

int AudioInput::getChannelCount()
{
	if (m_pAudioCodecCtx)
		return m_pAudioCodecCtx->channels;
	return 0;
}

AVSampleFormat AudioInput::getSampleFmt()
{
	if (m_pAudioCodecCtx)
		return m_pAudioCodecCtx->sample_fmt;
	return AV_SAMPLE_FMT_NONE;
}

/**
* Convert an error code into a text message.
* @param error Error code to be converted
* @return Corresponding error text (not thread-safe)
*/
const char *get_error_text(const int error)
{
	static char error_buffer[255];
	av_strerror(error, error_buffer, sizeof(error_buffer));
	return error_buffer;
}


/** Initialize one audio frame for reading from the input file */
int init_input_frame(AVFrame **frame)
{
	if (!(*frame = av_frame_alloc())) {
		fprintf(stderr, "Could not allocate input frame\n");
		return AVERROR(ENOMEM);
	}
	return 0;
}

/** Initialize one data packet for reading or writing. */
void init_packet(AVPacket *packet)
{
	av_init_packet(packet);
	/** Set the packet data and size so that it is recognized as being empty. */
	packet->data = NULL;
	packet->size = 0;
}

/** Decode one audio frame from the input file. */
int decode_audio_frame(AVFrame *frame,
	AVFormatContext *input_format_context,
	AVCodecContext *input_codec_context,
	int *data_present, int *finished)
{
	/** Packet used for temporary storage. */
	AVPacket input_packet;
	int error;
	init_packet(&input_packet);

	/** Read one audio frame from the input file into a temporary packet. */
	if ((error = av_read_frame(input_format_context, &input_packet)) < 0) {
		/** If we are at the end of the file, flush the decoder below. */
		if (error == AVERROR_EOF)
			*finished = 1;
		else {
			fprintf(stderr, "Could not read frame (error '%s')\n",
				get_error_text(error));
			return error;
		}
	}

	/**
	* Decode the audio frame stored in the temporary packet.
	* The input audio stream decoder is used to do this.
	* If we are at the end of the file, pass an empty packet to the decoder
	* to flush it.
	*/
	if ((error = avcodec_decode_audio4(input_codec_context, frame,
		data_present, &input_packet)) < 0) {
		fprintf(stderr, "Could not decode frame (error '%s')\n",
			get_error_text(error));
		av_packet_unref(&input_packet);
		return error;
	}

	/**
	* If the decoder has not been flushed completely, we are not finished,
	* so that this function has to be called again.
	*/
	if (*finished && *data_present)
		*finished = 0;
	av_packet_unref(&input_packet);
	return 0;
}

/******************** AudioMicInput Class **********************/

AudioMicInput::AudioMicInput()
{
	/** Register all codecs and formats so that they can be used. */
	av_register_all();
	avdevice_register_all();
}

AudioMicInput::~AudioMicInput()
{

}

/** Open an input file and the required decoder. */
int AudioMicInput::open(const char *filename)
{
	m_pFormatCtx = NULL;
	m_pAudioCodecCtx = NULL;

	if (strcmp(filename, "") == 0)
		return -1;

	/** Temporary storage of the input samples of the frame read from the file. */
	m_pAudioFrame = NULL;
	/** Initialize temporary storage for one input frame. */
	if (init_input_frame(&m_pAudioFrame))
		return -1;

	m_pTransferAudioFrame = av_frame_alloc();

	AVCodec *input_codec;
	int error;

	//************ Start **************//
	AVInputFormat* inputFormat = av_find_input_format("dshow");
	m_pFormatCtx = avformat_alloc_context();
	//************ End **************//


	/** Open the input file to read from it. */
	AVDictionary        *tOptions = NULL;
	av_dict_set(&tOptions, "rtbufsize", "100000000", 0);
	if ((error = avformat_open_input(&m_pFormatCtx, filename, inputFormat,
		&tOptions)) < 0) {
		fprintf(stderr, "Could not open input file '%s' (error '%s')\n",
			filename, get_error_text(error));
		m_pFormatCtx = NULL;
		return error;
	}

	/** Get information on the input file (number of streams etc.). */
	if ((error = avformat_find_stream_info(m_pFormatCtx, NULL)) < 0) {
		fprintf(stderr, "Could not open find stream info (error '%s')\n",
			get_error_text(error));
		avformat_close_input(&m_pFormatCtx);
		return error;
	}

	/** Make sure that there is only one stream in the input file. */
	if (m_pFormatCtx->nb_streams != 1) {
		fprintf(stderr, "Expected one audio input stream, but found %d\n",
			m_pFormatCtx->nb_streams);
		avformat_close_input(&m_pFormatCtx);
		return AVERROR_EXIT;
	}

	/** Find a decoder for the audio stream. */
	if (!(input_codec = avcodec_find_decoder(m_pFormatCtx->streams[0]->codec->codec_id))) {
		fprintf(stderr, "Could not find input codec\n");
		avformat_close_input(&m_pFormatCtx);
		return AVERROR_EXIT;
	}

	/** Open the decoder for the audio stream to use it later. */
	if ((error = avcodec_open2(m_pFormatCtx->streams[0]->codec,
		input_codec, NULL)) < 0) {
		fprintf(stderr, "Could not open input codec (error '%s')\n",
			get_error_text(error));
		avformat_close_input(&m_pFormatCtx);
		return error;
	}

	/** Save the decoder context for easier access later. */
	m_pAudioCodecCtx = m_pFormatCtx->streams[0]->codec;

	return 0;
}

void AudioMicInput::close()
{
	if (m_pAudioFrame) av_frame_free(&m_pAudioFrame);

	if (m_pTransferAudioFrame) av_frame_free(&m_pTransferAudioFrame);

	if (m_pAudioCodecCtx)
		avcodec_close(m_pAudioCodecCtx);

	if (m_pFormatCtx)
		avformat_close_input(&m_pFormatCtx);
}

int AudioMicInput::read(int *data_present, int *finished)
{
	if (m_pFormatCtx == NULL)
		return 1;
	/** Decode one frame worth of audio samples. */
	if (decode_audio_frame(m_pAudioFrame, m_pFormatCtx,
		m_pAudioCodecCtx, data_present, finished))
		return 1;

	return 0;
}

AVFrame *AudioMicInput::getAudioFrame()
{
	/*if (m_pAudioFrame == NULL || m_pAudioFrame->channels == 0)	return NULL;

	for (int i = 0; i < m_pAudioFrame->channels; i++) {
		if (m_pAudioFrame->extended_data[i] == NULL)
			m_pAudioFrame->channels--;
	}
	m_pAudioFrame->format = m_pAudioCodecCtx->sample_fmt;

	return m_pAudioFrame;*/
#if 0	// In this case, memory leak. (After av_frame_get_buffer, there is no release memory code.)
	// copy left channel to the output stream
	m_pTransferAudioFrame->format = m_pAudioFrame->format;
	m_pTransferAudioFrame->sample_rate = m_pAudioFrame->sample_rate;
	m_pTransferAudioFrame->nb_samples = m_pAudioFrame->nb_samples;
	m_pTransferAudioFrame->channels = 2;
	m_pTransferAudioFrame->channel_layout = av_get_default_channel_layout(m_pTransferAudioFrame->channels);

	av_frame_get_buffer(m_pTransferAudioFrame, 0);
	//memcpy(m_pTransferAudioFrame->extended_data[0], m_pAudioFrame->extended_data[0], m_pAudioFrame->nb_samples * av_get_bytes_per_sample(m_pAudioCodecCtx->sample_fmt));
	memcpy(m_pTransferAudioFrame->extended_data[0], m_pAudioFrame->extended_data[0], m_pAudioFrame->linesize[0]);

	return m_pTransferAudioFrame;
#else
	m_pAudioFrame->channels = 2;
	return m_pAudioFrame;
#endif
}