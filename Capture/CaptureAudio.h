#ifndef CAPTUREAUDIO_H
#define CAPTUREAUDIO_H

#pragma once

extern "C" {
#include "libswscale/swscale.h"
#include "libavdevice/avdevice.h"
#include "libavutil/avutil.h"
#include "libavutil/fifo.h"

#include "libavformat/avformat.h"
#include "libavformat/avio.h"

#include "libavcodec/avcodec.h"

#include "libavutil/audio_fifo.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/frame.h"
#include "libavutil/opt.h"

#include "libswresample/swresample.h"
}

#include <string>
#include <deque>

//#include <boost/thread.hpp>

#include <windows.h>
#include <vfw.h>

#include <stdio.h>
#include <tchar.h>

#include <stdlib.h>
#include <string.h>
#include <math.h>

/** The output bit rate in kbit/s */
#define OUTPUT_BIT_RATE 96000
/** The number of output channels */
#define OUTPUT_CHANNELS 2

class AudioInput
{
public:
	virtual int getSampleRate();
	virtual int getChannelCount();
	virtual AVSampleFormat getSampleFmt();

	AVCodecContext *m_pAudioCodecCtx;
	AVFrame *m_pAudioFrame;			// audio frame from device
	AVFrame *m_pTransferAudioFrame; // audio frame for transfer to the output
};

class AudioMicInput : public AudioInput
{
	// Constructor & Destructor
public:
	AudioMicInput();
	virtual ~AudioMicInput();

	// Member Functions
public:
	int open(const char *filename);
	void close();

	int read(int *data_present, int *finished);

	//AVCodecContext *getCodecContext();
	AVFrame *getAudioFrame();


	AVFormatContext *m_pFormatCtx;
};

const char *get_error_text(const int error);

void init_packet(AVPacket *packet);

#endif //CAPTUREAUDIO_H