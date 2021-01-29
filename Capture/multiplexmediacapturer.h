///
//
// LibSourcey
// Copyright (c) 2005, Sourcey <https://sourcey.com>
//
// SPDX-License-Identifier:	LGPL-2.1+
//
/// @addtogroup webrtc
/// @{


#ifndef SCY_WebRTC_webRTC_MultiplexMediaCapturer_H
#define SCY_WebRTC_webRTC_MultiplexMediaCapturer_H


#include "scy/base.h"

//#ifdef HAVE_FFMPEG

#include "scy/av/av.h"
#include "scy/av/mediacapture.h"
#include "scy/av/packet.h"
#include "scy/webrtc/audiopacketmodule.h"
#include "scy/webrtc/videopacketsource.h"

#include "webrtc/api/peerconnectioninterface.h"


namespace scy {
namespace av {

/// This class implements a cross platform audio, video, screen and
/// video file capturer.
class AV_API webRTC_MediaCapture : public MediaCapture
{
public:
	typedef std::shared_ptr<webRTC_MediaCapture> webRTC_Ptr;

	webRTC_MediaCapture();
	virtual ~webRTC_MediaCapture();

	virtual void start() override;
	virtual void stop() override;
	virtual void run() override;
	void webRTC_makeOneFrameStream(AVFrame *oneFraame);

protected:
	virtual void openStream(const std::string& filename, AVInputFormat* inputFormat, AVDictionary** formatParams);

	void webRTC_emit(IPacket& packet);

	bool _starting;
};
}

namespace wrtc {


class webRTC_MultiplexMediaCapturer
{
public:
	webRTC_MultiplexMediaCapturer();
    ~webRTC_MultiplexMediaCapturer();

    void openFile(const std::string& file, bool loop = true);

    void addMediaTracks(webrtc::PeerConnectionFactoryInterface* factory,
                        webrtc::MediaStreamInterface* stream);

    void start();
    void stop();

	rtc::scoped_refptr<AudioPacketModule> getAudioModule();
	av::webRTC_MediaCapture::webRTC_Ptr getVideoCapture_webPtr();
	VideoPacketSource* createVideoSource();

protected:
    PacketStream _stream;
    av::webRTC_MediaCapture::webRTC_Ptr _videoCapture;
    rtc::scoped_refptr<AudioPacketModule> _audioModule;
};

} } // namespace scy::wrtc


//#endif // HAVE_FFMPEG
#endif

