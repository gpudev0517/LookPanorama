<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <!-- Origin Trial Token, feature = WebVR, origin = https://webvr.info, expires = 2017-07-18 -->
    <meta http-equiv="origin-trial" data-feature="WebVR" data-expires="2017-07-18" content="AmcZQ9UIoWBHR8Q3p/7GHNVux0D2rIfqemrRFSPBaSBb8BViQVPqx3zd8i4mMIf8J70oRhdQqCvafYfBEGAonwkAAABJeyJvcmlnaW4iOiJodHRwczovL3dlYnZyLmluZm86NDQzIiwiZmVhdHVyZSI6IldlYlZSIiwiZXhwaXJ5IjoxNTAwMzM2MDAwfQ==">
    <!-- Origin Trial Token, feature = WebVR (For Chrome M59+), origin = https://webvr.info, expires = 2017-07-21 -->
    <meta http-equiv="origin-trial" data-feature="WebVR (For Chrome M59+)" data-expires="2017-07-21" content="AtJLsI9hT0/XyPU7DEDDxER7jyMU1oNeMk4diF9djsHCkwjulNzizKykf+CKW11B6+0ABoazxbd13jMxBvnUTQIAAABfeyJvcmlnaW4iOiJodHRwczovL3dlYnZyLmluZm86NDQzIiwiZmVhdHVyZSI6IldlYlZSMS4xIiwiZXhwaXJ5IjoxNTAwNjc3MDE3LCJpc1N1YmRvbWFpbiI6dHJ1ZX0=">

    <title>PanoOne</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css" integrity="sha384-rwoIResjU2yc3z8GV/NPeZWAv56rSmLldC3R/AZzGRnGxQQKnKkoFVhFQhNUwEyJ" crossorigin="anonymous">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css">
    <link href="https://gitcdn.github.io/bootstrap-toggle/2.2.2/css/bootstrap-toggle.min.css" rel="stylesheet">
    <link rel="stylesheet" href="css/app.css">
    
    <script src="https://code.jquery.com/jquery-3.1.1.slim.min.js" integrity="sha384-A7FZj7v+d/sdmMqp/nOQwliLvUsJfDHW+k9Omg/a/EheAdgtzNs3hpfag6Ed950n" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js" integrity="sha384-DztdAPBWPRXSA/3eYEEUWrWCy7G5KFbe8fFjk5JAIxUYHKkDx6Qin1DkWx51bBrb" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js" integrity="sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn" crossorigin="anonymous"></script>
    <script src="https://gitcdn.github.io/bootstrap-toggle/2.2.2/js/bootstrap-toggle.min.js"></script>
    
    <script>
      var WebVRConfig = {
        // Prevents the polyfill from initializing automatically.
        DEFER_INITIALIZATION: true,
        // Ensures the polyfill is always active when initialized, even if the
        // native API is available. This is probably NOT what most pages want.
        POLYFILL_MODE: "ALWAYS",
        // Polyfill optimizations
        DIRTY_SUBMIT_FRAME_BINDINGS: true,
        BUFFER_SCALE: 0.75,
      };
    </script>
    <script src="js/third-party/webvr-polyfill.js"></script>
    <script src="js/third-party/wglu/wglu-url.js"></script>
    <script>
      // Dynamically turn the polyfill on if requested by the query args.
      if (WGLUUrl.getBool('polyfill', false)) {
        InitializeWebVRPolyfill();
      } else {
        // Shim for migration from older version of WebVR. Shouldn't be necessary for very long.
        InitializeSpecShim();
      }
    </script>
    <!-- End sample polyfill enabling logic -->
    <script src="js/third-party/gl-matrix-min.js"></script>
    <script src="js/third-party/wglu/wglu-program.js"></script>
    <script src="js/third-party/wglu/wglu-stats.js"></script>
    <script src="js/vr-panorama.js"></script>
    <script src="js/vr-samples-util.js"></script>
    <script src="js/startVR.js"></script>
    <script src="js/hls.min.js"></script>
    <script src="js/app.js"></script>
  </head>
  <body>
    <div class="instructions-overlay">
      <div class="instructions">
        <!--<img src="images/look.png" alt="logo">-->
        <div style="margin-top: 250px;">
          <h2>LookVR Viewer</h2>
          <!--<p>This dance was captured using a VR headset and controllers.</p>-->
        </div>
        <div style="display: inline-block; padding:20px;">
          <label for="video_addr" id="server-address-note">You need to input valid server address here.</label>
          <div class="form-section">
            <label for="video_addr" class="">Server :</label>
            <input type="text" class="form-control video-address" id="video_addr" placeholder="Eg. localhost:1935" value="localhost:1935">
          </div>
          <div class="form-section">
            <label for="video_addr">Application: </label>
            <input type="text" class="form-control video-address" id="video_addr_sub_1" placeholder="Eg. live" value="live">
          </div>
          <div class="form-section">
            <label for="video_addr">Stream :</label>
            <input type="text" class="form-control video-address" id="video_addr_sub_2" placeholder="Eg. myStream" value="myStream">
          </div>
          <div class="form-section">
            <label for="video_addr">Full URL :</label>
            <input type="text" class="form-control video-full-url" disabled value="http://localhost:1935/live/myStream">
          </div>
          <div class="form-section">
            <label for="headset_status">Oculus :</label>
            <input type="text" class="form-control headset-connected" disabled value="None">
          </div>
          <input type="checkbox" checked data-toggle="toggle" data-width="100" name="video_mode" class="form-control video_mode" data-on="Mono Mode" data-off="Stereo Mode" data-width="400" data-offstyle="info">
        </div>
        <button class="start-button btn-set-video" ><span class="">START</span></button>
      </div>
    </div>
    <canvas id="webgl-canvas" class="hidden"></canvas>
    <video id="Wowza_video" class="hidden" width="500" height="500"></video>
    <span class="screen-top-left hidden">Connecting now...</span>
    <button type="button" class="btn btn-primary btn-modal-toggler">
      <i class="fa fa-close"></i>
    </button>
  </body>
</html>
