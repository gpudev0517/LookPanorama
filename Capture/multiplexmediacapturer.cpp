///
//
// LibSourcey
// Copyright (c) 2005, Sourcey <https://sourcey.com>
//
// SPDX-License-Identifier:	LGPL-2.1+
//
/// @addtogroup webrtc
/// @{


#include "multiplexmediacapturer.h"

//#ifdef HAVE_FFMPEG

#include "scy/av/audioresampler.h"
#include "scy/av/ffmpeg.h"
#include "scy/av/realtimepacketqueue.h"
#include "scy/av/videocontext.h"
#include "scy/filesystem.h"
#include "scy/logger.h"
#include "scy/webrtc/webrtc.h"
#include "webrtc/media/engine/webrtcvideocapturerfactory.h"
#include "webrtc/modules/video_capture/video_capture_factory.h"

using std::endl;

namespace scy {
namespace av {

webRTC_MediaCapture::webRTC_MediaCapture()
	: MediaCapture()
{
	_starting = false;
	initializeFFmpeg();
}

webRTC_MediaCapture::~webRTC_MediaCapture()
{
	close();
	uninitializeFFmpeg();
}

void webRTC_MediaCapture::openStream(const std::string& filename, AVInputFormat* inputFormat, AVDictionary** formatParams)
{
	LTrace("Opening stream: ", filename)

		if (_formatCtx)
			throw std::runtime_error("Capture has already been initialized");

	if (avformat_open_input(&_formatCtx, filename.c_str(), inputFormat, formatParams) < 0)
		throw std::runtime_error("Cannot open the media source: " + filename);

	// _formatCtx->max_analyze_duration = 0;
	if (avformat_find_stream_info(_formatCtx, nullptr) < 0)
		throw std::runtime_error("Cannot find stream information: " + filename);

	av_dump_format(_formatCtx, 0, filename.c_str(), 0);

	//avformat_alloc_output_context2(&_formatCtx, NULL, "h264", filename.c_str());

	for (unsigned i = 0; i < _formatCtx->nb_streams; i++) {
		auto stream = _formatCtx->streams[i];
		auto codec = stream->codec;
		if (!_video && codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			_video = new VideoDecoder(stream);
			_video->emitter.attach(packetSlot(this, &webRTC_MediaCapture::webRTC_emit));
			_video->create();
			_video->open();
		}
		else if (!_audio && codec->codec_type == AVMEDIA_TYPE_AUDIO) {
			_audio = new AudioDecoder(stream);
			_audio->emitter.attach(packetSlot(this, &webRTC_MediaCapture::webRTC_emit));
			_audio->create();
			_audio->open();
		}
	}

	if (!_video && !_audio)
		throw std::runtime_error("Cannot find a valid media stream: " + filename);
}

void webRTC_MediaCapture::webRTC_emit(IPacket& packet)
{
	LTrace("Emit: ", packet.size())

		emitter.emit(packet);
}

void webRTC_MediaCapture::run()
{
	LTrace("Running")

	try {
		int res;
		AVPacket ipacket;
		av_init_packet(&ipacket);

		// Looping variables
		int64_t videoPtsOffset = 0;
		int64_t audioPtsOffset = 0;

		// Realtime variables
		int64_t lastTimestamp = time::hrtime();
		int64_t frameInterval = _video ? fpsToInterval(int(_video->iparams.fps)) : 0;

		// Reset the stream back to the beginning when looping is enabled
		if (_looping) {
			LTrace("Looping")
				for (unsigned i = 0; i < _formatCtx->nb_streams; i++) {
					if (avformat_seek_file(_formatCtx, i, 0, 0, 0, AVSEEK_FLAG_FRAME) < 0) {
						throw std::runtime_error("Cannot reset media stream");
					}
				}
		}

		// Read input packets until complete
		while ((res = av_read_frame(_formatCtx, &ipacket)) >= 0) {
			STrace << "Read frame: "
				<< "pts=" << ipacket.pts << ", "
				<< "dts=" << ipacket.dts << endl;

			if (_stopping)
				break;

			if (_video && ipacket.stream_index == _video->stream->index) {

				// Set the PTS offset when looping
				if (_looping) {
					if (ipacket.pts == 0 && _video->pts > 0)
						videoPtsOffset = _video->pts;
					ipacket.pts += videoPtsOffset;
				}

				// Decode and emit
				if (_video->decode(ipacket)) {
					STrace << "Decoded video: "
						<< "time=" << _video->time << ", "
						<< "pts=" << _video->pts << endl;
				}

				// Pause the input stream in realtime mode if the
				// decoder is working too fast
				if (_realtime) {
					auto nsdelay = frameInterval - (time::hrtime() - lastTimestamp);
					// LDebug("Sleep delay: ", nsdelay, ", ", (time::hrtime() - lastTimestamp), ", ", frameInterval)
					std::this_thread::sleep_for(std::chrono::nanoseconds(nsdelay));
					lastTimestamp = time::hrtime();
				}
			}
			else if (_audio && ipacket.stream_index == _audio->stream->index) {

				// Set the PTS offset when looping
				if (_looping) {
					if (ipacket.pts == 0 && _audio->pts > 0)
						videoPtsOffset = _audio->pts;
					ipacket.pts += audioPtsOffset;
				}

				// Decode and emit
				if (_audio->decode(ipacket)) {
					STrace << "Decoded Audio: "
						<< "time=" << _audio->time << ", "
						<< "pts=" << _audio->pts << endl;
				}
			}

			av_packet_unref(&ipacket);
		}

		// Flush remaining packets
		if (!_stopping && res < 0) {
			if (_video)
				_video->flush();
			if (_audio)
				_audio->flush();
		}

		// End of file or error
		LTrace("Decoder EOF: ", res)
	}
	catch (std::exception& exc) {
		_error = exc.what();
		LError("Decoder Error: ", _error)
	}
	catch (...) {
		_error = "Unknown Error";
		LError("Unknown Error")
	}

	if (_stopping || !_looping) {
		LTrace("Exiting")
			_stopping = true;
		Closing.emit();
	}
}

inline void emitPacket(VideoDecoder* dec, AVFrame* frame);

void webRTC_MediaCapture::webRTC_makeOneFrameStream(AVFrame *oneFraame)
{
	if ((_video || _audio) && _starting && !_stopping) {
		emitPacket(_video, oneFraame); //, frame, opacket, &ptsSeconds
	}
}

void webRTC_MediaCapture::start()
{
	LTrace("Starting")

	std::lock_guard<std::mutex> guard(_mutex);
	assert(_video || _audio);

	if ((_video || _audio) && !_starting) {
		_starting = true;
		_stopping = false;
		//_thread.start(std::bind(&webRTC_MediaCapture::run, this));
	}
}

void webRTC_MediaCapture::stop()
{
	LTrace("Stopping")

	std::lock_guard<std::mutex> guard(_mutex);

	_starting = false;
	_stopping = true;
}

}

namespace wrtc {


	webRTC_MultiplexMediaCapturer::webRTC_MultiplexMediaCapturer()
    : _videoCapture(std::make_shared<av::webRTC_MediaCapture>())
    , _audioModule(AudioPacketModule::Create())
{
    _stream.attachSource(_videoCapture, true);
    // _stream.attach(std::make_shared<av::RealtimePacketQueue<av::MediaPacket>>(0), 5);
    // _stream.attach(std::make_shared<av::RealtimePacketQueue<av::PlanarVideoPacket>>(0), 5);
    _stream.emitter += packetSlot(_audioModule.get(), &AudioPacketModule::onAudioCaptured);
}


	webRTC_MultiplexMediaCapturer::~webRTC_MultiplexMediaCapturer()
{
}


void webRTC_MultiplexMediaCapturer::openFile(const std::string& file, bool loop)
{
    // Open the capture file
    _videoCapture->setLoopInput(loop);
	_videoCapture->setRealtimePlayback(false);
	//_videoCapture->setRealtimePlayback(true);
    _videoCapture->openFile(file);

    // Set the output settings
    if (_videoCapture->audio()) {
        _videoCapture->audio()->oparams.sampleFmt = "s16";
        _videoCapture->audio()->oparams.sampleRate = 44000;
        _videoCapture->audio()->oparams.channels = 2;
        _videoCapture->audio()->recreateResampler();
        // _videoCapture->audio()->resampler->maxNumSamples = 440;
        // _videoCapture->audio()->resampler->variableOutput = false;
    }

    // Convert to yuv420p for WebRTC compatability
    if (_videoCapture->video()) {
        _videoCapture->video()->oparams.pixelFmt = "yuv420p"; // nv12
        // _videoCapture->video()->oparams.width = capture_format.width;
        // _videoCapture->video()->oparams.height = capture_format.height;
    }
}


VideoPacketSource* webRTC_MultiplexMediaCapturer::createVideoSource()
{
    assert(_videoCapture->video());
    auto oparams = _videoCapture->video()->oparams;
    auto source = new VideoPacketSource(oparams.width, oparams.height,
                                        oparams.fps, cricket::FOURCC_I420);
    source->setPacketSource(&_stream.emitter); // nullified on VideoPacketSource::Stop
    return source;
}


rtc::scoped_refptr<AudioPacketModule> webRTC_MultiplexMediaCapturer::getAudioModule()
{
    return _audioModule;
}

av::webRTC_MediaCapture::webRTC_Ptr webRTC_MultiplexMediaCapturer::getVideoCapture_webPtr()
{
	return _videoCapture;
}


void webRTC_MultiplexMediaCapturer::addMediaTracks(
    webrtc::PeerConnectionFactoryInterface* factory,
    webrtc::MediaStreamInterface* stream)
{
    // This capturer is multicast, meaning it can be used as the source
    // for multiple Peer objects.
    //
    // KLUDGE: Pixel format conversion should happen on the
    // VideoPacketSource rather than on the decoder becasue different
    // peers may request different optimal output video sizes.

    // Create and add the audio stream
    if (_videoCapture->audio()) {
        stream->AddTrack(factory->CreateAudioTrack(
            kAudioLabel, factory->CreateAudioSource(nullptr)));
    }

    // Create and add the video stream
    if (_videoCapture->video()) {
        stream->AddTrack(factory->CreateVideoTrack(
            kVideoLabel, factory->CreateVideoSource(createVideoSource(), nullptr)));
    }

    // Default WebRTC video stream for testing
    // stream->AddTrack(factory->CreateVideoTrack(
    //     kVideoLabel, factory->CreateVideoSource(openVideoDefaultWebRtcCaptureDevice(), nullptr)));
}


void webRTC_MultiplexMediaCapturer::start()
{
    _stream.start();
}


void webRTC_MultiplexMediaCapturer::stop()
{
    _stream.stop();
}


} } // namespace scy::wrtc


//#endif // HAVE_FFMPEG
