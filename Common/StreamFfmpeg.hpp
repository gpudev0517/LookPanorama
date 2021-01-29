
#ifndef __Stream_FFMPEG__
#define __Stream_FFMPEG__

#include "Capture.h"
#include "BaseFfmpeg.hpp"
#include <QMap>
#include <QMutex>

extern "C" {
#include "libavutil/audio_fifo.h"
}


typedef struct OutputStream 
{
	OutputStream()
	{
		st = NULL;
		frame = NULL;
		tmp_frame = NULL;
		t = 0;
		tincr = 0;
		tincr2 = 0;
		sws_ctx = NULL;
		swr_ctx = NULL;
		Initialize();
	}
	void Initialize()
	{
		next_pts = 0;
		samples_count = 0;
		now_time = -1;
		begin_offset = 0;
		audio_synced = false;
	}
	AVStream *st;

	/* pts of the next frame that will be generated */
	int64_t next_pts;
	int samples_count;

	AVFrame *frame;
	AVFrame *tmp_frame;

	float t, tincr, tincr2;

	struct SwsContext *sws_ctx;
	struct SwrContext *swr_ctx;

	int64_t now_time;
	int64_t begin_offset; // for split files, audio can have some offset for the first packet.
	bool audio_synced;
} OutputStream;





class StreamFfmpeg : public virtual BaseFfmpeg
{
private:
	
	AVFormatContext *iFormatContext_;

	int videoIndex;
	int frameIndex;
	int64_t startTime;

	int errVal;

	OutputStream mOutputVideoStream;
	OutputStream mOutputAudioStream;

	AVCodec      *mAudioCodec;
	AVCodec      *mVideoCodec;

	SwrContext *resample_context;
	AVAudioFifo *fifo;
	AVSampleFormat m_inSampleFmt;

	int  mHaveVideo;
	int  mHaveAudio;
	bool mEncodeVideo;
	bool mEncodeAudio;

	bool mInitialized;
	QMutex streamMutex;
	D360::Capture::CaptureDomain m_capType;
public:
	StreamFfmpeg();
	virtual ~StreamFfmpeg() { };

	int Initialize( std::string fileName, std::string outFileName ) ;
	int initializeContextFromFile( std::string fileName );
	int addStreamsFromInputContext();
	int WriteFrame();

	int createVideoBuffers(unsigned int width, unsigned int height);
	int addVideoStream(unsigned int width, unsigned int height);
	int WriteFrame(uint8_t* input, int frame_index, int width, int height);


	int Initialize(std::string outFileName, int width, int height, int fps, int channelCount, 
		AVSampleFormat sampleFmt, int srcSampleRate, int sampleRate, int audioLag, bool toFile, int videoCodec, int audioCodec, bool bLidar, int crf = 23);
	int Initialize_webRTC(std::string outFileName, int width, int height, int fps, int channelCount,
		AVSampleFormat sampleFmt, int srcSampleRate, int sampleRate, int audioLag, bool toFile, int videoCodec, int audioCodec, int crf = 23);
	int Init_Audio_Resampler(int channelCount, AVSampleFormat sampleFmt, int sampleRate);
	int Init_Audio_Fifo();

	int StreamFrame( uint8_t* input, int frame_index, int width, int height, bool sync );
	int StreamLiDARFrame( uint8_t* input, int width, int height );
	AVFrame* webRTC_makeStreamFrame(uint8_t* input, int frame_index, int width, int height, bool sync);
	int StoreAudioFrame(void* audioFrame);
	int StoreAudioFrame(QMap<int, void*> audioFrame);
	int StreamAudio();
	int StreamRemainingSamples();

	// Audio resampler, FIFO


	int error();
	int Close();

	void setCaptureType(D360::Capture::CaptureDomain capType) { m_capType = capType; }

private:
	void Clean();

	int load_encode_and_write();
	int encode_audio_frame(AVFrame *frame,
		int *data_present);

	/** Global timestamp for the audio frames */
	int64_t pts;

	int audioLag;
	int videoFps;

	QMutex streamThread;
};

#endif // __Stream_FFMPEG__