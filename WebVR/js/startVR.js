/* global mat4, vec3, VRPanorama, WGLUStats, VRSamplesUtil */
function startVR() {
    "use strict";
    if(!$('.screen-top-left').hasClass('hidden'))
        $('.screen-top-left').addClass('hidden');
    var vrDisplay = null;
    var frameData = null;
    var isStereo = false;
    var projectionMat = mat4.create();
    var poseMat = mat4.create();
    var viewMat = mat4.create();
    var vrPresentButton = null;
    // ================================================================
    // WebGL and WebAudio scene setup. This code is not WebVR specific.
    // ================================================================
    // WebGL setup.
    var webglCanvas = document.getElementById("webgl-canvas");
    var gl = null;
    var panorama = null;
    var stats = null;
    function init (preserveDrawingBuffer) {
        var glAttribs = {
            alpha: false,
            antialias: false,
            preserveDrawingBuffer: preserveDrawingBuffer
        };
        gl = webglCanvas.getContext("webgl", glAttribs);
        if (!gl) {
            gl = webglCanvas.getContext("experimental-webgl", glAttribs);
            if (!gl) {
            VRSamplesUtil.addError("Your browser does not support WebGL.");
            return;
            }
        }
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.CULL_FACE);
        panorama = new VRPanorama(gl);
        var defaultVideo = "media/textures/pano.webm";
        var videoUrl = WGLUUrl.getString(
            'videoUrl', defaultVideo);
        if(videoMode == "mode_mono")
            panorama.setStereo(false);
        else
            panorama.setStereo(true);
        panorama.setVideo('Wowza_video').catch(
            function(error) {
                if (error instanceof MediaError && error.code === 4) {
                panorama.setVideo('Wowza_video');
                }
            });
        var enablePerformanceMonitoring = WGLUUrl.getBool(
            'enablePerformanceMonitoring', false);
        stats = new WGLUStats(gl, enablePerformanceMonitoring);
        // Wait until we have a WebGL context to resize and start rendering.
        window.addEventListener("resize", onResize, false);
        onResize();
        window.requestAnimationFrame(onAnimationFrame);
    }
    // ================================
    // WebVR-specific code begins here.
    // ================================
    function onVRRequestPresent () {
    vrDisplay.requestPresent([{ source: webglCanvas }]).then(function () {
    }, function (err) {
        var errMsg = "requestPresent failed.";
        if (err && err.message) {
        errMsg += "<br/>" + err.message
        }
        VRSamplesUtil.addError(errMsg, 2000);
    });
    }
    function onVRExitPresent () {
    if (!vrDisplay.isPresenting)
        return;
    vrDisplay.exitPresent().then(function () {
    }, function () {
        VRSamplesUtil.addError("exitPresent failed.", 2000);
    });
    }
    function onVRPresentChange () {
    onResize();
    if (vrDisplay.isPresenting) {
        if (vrDisplay.capabilities.hasExternalDisplay) {
        VRSamplesUtil.removeButton(vrPresentButton);
        vrPresentButton = VRSamplesUtil.addButton("Exit VR", "E", "media/icons/cardboard64.png", onVRExitPresent);
        }
    } else {
        if (vrDisplay.capabilities.hasExternalDisplay) {
        VRSamplesUtil.removeButton(vrPresentButton);
        vrPresentButton = VRSamplesUtil.addButton("Enter VR", "E", "media/icons/cardboard64.png", onVRRequestPresent);
        }
    }
    }
    if (navigator.getVRDisplays) {
        frameData = new VRFrameData();
        navigator.getVRDisplays().then(function (displays) {
            if (displays.length > 0) {
            vrDisplay = displays[displays.length - 1];
            vrDisplay.depthNear = 0.1;
            vrDisplay.depthFar = 1024.0;
            init(true);
            if (vrDisplay.capabilities.canPresent)
                vrPresentButton = VRSamplesUtil.addButton("Enter VR", "E", "media/icons/cardboard64.png", onVRRequestPresent);
            // For the benefit of automated testing. Safe to ignore.
            if (vrDisplay.capabilities.canPresent && WGLUUrl.getBool('canvasClickPresents', false))
                webglCanvas.addEventListener("click", onVRRequestPresent, false);
            window.addEventListener('vrdisplaypresentchange', onVRPresentChange, false);
            window.addEventListener('vrdisplayactivate', onVRRequestPresent, false);
            window.addEventListener('vrdisplaydeactivate', onVRExitPresent, false);
            } else {
            init(false);
            VRSamplesUtil.addInfo("WebVR supported, but no VRDisplays found.", 3000);
            }
        });
    } else if (navigator.getVRDevices) {
        init(false);
        VRSamplesUtil.addError("Your browser supports WebVR but not the latest version. See <a href='http://webvr.info'>webvr.info</a> for more info.");
    } else {
        init(false);
        VRSamplesUtil.addError("Your browser does not support WebVR. See <a href='http://webvr.info'>webvr.info</a> for assistance.");
    }
    function onResize () {
        if (vrDisplay && vrDisplay.isPresenting) {
            var leftEye = vrDisplay.getEyeParameters("left");
            var rightEye = vrDisplay.getEyeParameters("right");
            webglCanvas.width = Math.max(leftEye.renderWidth, rightEye.renderWidth) * 2;
            webglCanvas.height = Math.max(leftEye.renderHeight, rightEye.renderHeight);
        } else {
            webglCanvas.width = webglCanvas.offsetWidth * window.devicePixelRatio;
            webglCanvas.height = webglCanvas.offsetHeight * window.devicePixelRatio;
        }
    }
    function getPoseMatrix (out, pose) {
        // When rendering a panorama ignore the pose position. You want the
        // users head to stay centered at all times. This would be terrible
        // advice for any other type of VR scene, by the way!
        var orientation = pose.orientation;
        if (!orientation) { orientation = [0, 0, 0, 1]; }
        mat4.fromQuat(out, orientation);
        mat4.invert(out, out);
    }
    function onAnimationFrame (t) {
        stats.begin();
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        panorama.setRerendering(true);
        if (vrDisplay) {
            vrDisplay.requestAnimationFrame(onAnimationFrame);
            vrDisplay.getFrameData(frameData);
            // FYI: When rendering a panorama do NOT use view matricies directly!
            // That will make the viewer feel like their head is trapped in a tiny
            // ball, which is usually not the desired effect. Instead, render both
            // eyes from a single viewpoint.
            getPoseMatrix(viewMat, frameData.pose);
            if (vrDisplay.isPresenting) {
                gl.viewport(0, 0, webglCanvas.width * 0.5, webglCanvas.height);
                panorama.render(frameData.leftProjectionMatrix, viewMat, 0/*=left*/);
                gl.viewport(webglCanvas.width * 0.5, 0, webglCanvas.width * 0.5, webglCanvas.height);
                panorama.render(frameData.rightProjectionMatrix, viewMat, 1/*=right*/);
                vrDisplay.submitFrame();
            } else {
                gl.viewport(0, 0, webglCanvas.width, webglCanvas.height);
                mat4.perspective(projectionMat, Math.PI*0.4, webglCanvas.width / webglCanvas.height, 0.1, 1024.0);
                panorama.render(projectionMat, viewMat, 2/*=mono*/);
                stats.renderOrtho();
            }
        } else {
            window.requestAnimationFrame(onAnimationFrame);
            // No VRDisplay found.
            gl.viewport(0, 0, webglCanvas.width, webglCanvas.height);
            mat4.perspective(projectionMat, Math.PI*0.4, webglCanvas.width / webglCanvas.height, 0.1, 1024.0);
            mat4.identity(viewMat);
            panorama.render(projectionMat, viewMat);
            stats.renderOrtho();
        }
        stats.end();
    }
}