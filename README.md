# README #

To build the application PanoOne, the following are the dependencies


1. Eigen
2. Zlib
3. Ffmpeg
4. OculusSDK 0.8.0


### What is this repository for? ###

* This is the repository for Panorama stitcher for custom Camera setup
* 2.0


### How do I get set up? ###

- Summary of set up
  - Download Dependency Packages
  - Compile or Build them if needed.
  - Configure CMake using the CMakeLists.txt
  - Compile and Build
- Configuration
  - You might get errors about platform for qt in debug version when running. Make platforms directory in Debug, and copy qwindowsd.dll there.
  - Copy External/Binary/PanoramaTools directory to the binary folder.
- Deployment instructions

### How to run? ###
- This project needs ini file for panoramic camera rig and output format.
  - List of sources (live camera, static image, or from video file)
  - Camera calibration information
  - Output target (Oculus Rift, video file or wowza server)
  - 3D Stereo information
    ```ini
    - [D360]
    3D=(true, false)
    3DOutput=(topdown,sidebyside)
    - [CameraCapture]
    3DCameraInputConfig=(stereo, global)
    cameraCalibFileLeft, cameraCalibFileRight
    - [Camera n] --> [LeftCamera n], [RightCamera n]
    ```
    
- We have some format for camera calibration file.
  - Camera calibration file has extension of PAC. 
  - You need to run ptGUI for camera calibration. Set the camera format as "Fullframe". This is the most common one, but certainly we only assume this format.
  - Copy rotation and camera interior parameters to PAC file.
  - If you see a vertically mirrored image, add this in the last of tag.
  - * <flipped value="True"/>

- Argument format.
  
  Format:
  `./Look3D.exe [--nogui] [--config] [--broadcast]`

  - `--nogui` : Run as console mode.
  - `--config`: path of l3d configuration file made from standalone mode.
  - `--broadcast`: Target streaming path.

  - Example:
    > --nogui --config D:/SampleData/liveFLIR/live_2camFLIR.l3d --broadcast udp://127.0.0.1:8051

- UI mode and server mode

  This project can be with two mode(UI mode and server mode)
  - UI mode

    UI mode is need for editing and saving l3d configuration file.
    No argument means that Project be run in UI mode.

    Format: `./Look3D.exe`

  - Server mode

    Server mode is need for sending paranomic image to Unity client.
    You have to run server mode after run client and connect.
    Server mode use a configuration file saved in UI mode.
    This mode exactly needs three argument.

    Format: `./Look3D.exe [--nogui] [--config] [--broadcast]`

  - l3d configuration file

    l3d file is zip file format, include ini file and weighmap files.
    You can edit l3d file manually by exchange file extention to zip( or rar).
  - Run Unity Client

    You have to run unity client and press connect button before run server mode.
    You can set server url and image size.
- Envirenment
  - Install Spinnaker sdk.

- Finish Program.
  - At first finish unity client.
  - Next finish server program.

### How does it work? ###
- (CPU) ffmpeg dshow capture --> (yuv420) -->
- (GPU) --> GLSLColorCvt_YUV2RGB --> (rgb) --> GLSLUnwarp --> (n:1) --> GLSLComposite --> (rgb) GLSLColorCvtRGB2YUV --> (yuv420) -->
- (CPU) --> ffmpeg streaming

### Configuration for webvr experience ###
- Download and run chromium (chromium_webvr_v1.1_win.zip) from following address.
  > https://webvr.info/get-chrome/
- Enter chrome://flags/#enable-webvr in the URL bar, and press "Enable".
- Run webvr page.

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

