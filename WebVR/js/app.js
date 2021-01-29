var serverURL = "";
var videoMode = "mode_mono";
var hls = null;
$(document).ready(function(){
    // if (!AFRAME.utils.isMobile() && AFRAME.utils.checkHeadsetConnected())
	// 	$('.headset-connected').val('Connected');
	// else
	// 	$('.headset-connected').val('None');
    $('.instructions').addClass('in');
    $('.video_mode').bootstrapToggle({
      width: '350'
    });
    $('.video-address').change(function(){
        serverURL = 'http://' + $('#video_addr').val();
        serverURL += '/' + $('#video_addr_sub_1').val();
        serverURL += '/' + $('#video_addr_sub_2').val();
        $('.video-full-url').val(serverURL);
    });
    $('.btn-set-video').click(function(){
        serverURL = 'http://' + $('#video_addr').val();
        serverURL += '/' + $('#video_addr_sub_1').val();
        serverURL += '/' + $('#video_addr_sub_2').val();
        if(serverURL.length < 7){
            if(!$('#server-address-note').hasClass('color-brown'))
                $('#server-address-note').addClass('color-brown');
            return;
        }
        else{
            if($('#server-address-note').hasClass('color-brown'))
                $('#server-address-note').removeClass('color-brown');
        }
        videoMode = $('input[name=video_mode]').prop('checked');
        if(videoMode == true)
            videoMode = "mode_mono";
        else
            videoMode = "mode_stereo";
        setVideo();
        $('.instructions-overlay').addClass('hidden');
        $('#webgl-canvas').removeClass('hidden');
    });
    $('.btn-modal-toggler').click(function(){
        location.reload();
    });
});
function setVideo(){
    if(Hls.isSupported()) {
        var video = document.getElementById('Wowza_video');
        function init(){
            if(hls != null){
                hls.destroy();
            }
            hls = new Hls();
            hls.loadSource(serverURL + '/playlist.m3u8');
            hls.attachMedia(video);
            hls.on(Hls.Events.MANIFEST_PARSED,function() {
                video.play();
                video.addEventListener('playing', function() {
                    startVR();
                });
            });
        }
        init();
        if($('.screen-top-left').hasClass('hidden'))
            $('.screen-top-left').removeClass('hidden');
    }
    // var stream = {
    //     url: serverURL + "/Manifest"
    // };
    // var video = document.getElementById('Wowza_video');
    // video.addEventListener('playing', function() {
    //     startVR();
    // });
    // var mediaPlayer = new MediaPlayer();
    // mediaPlayer.init(document.querySelector("#Wowza_video"));
    // mediaPlayer.load(stream);
    // if($('.screen-top-left').hasClass('hidden'))
    //     $('.screen-top-left').removeClass('hidden');
}