TEMPLATE = app

QT += qml quick widgets network opengl
CONFIG += c++11

SOURCES += \
    main.cpp \
    Capture/AudioThread.cpp \
    Capture/BannerThread.cpp \
    Capture/CameraView.cpp \
    Capture/Capture.cpp \
    Capture/CaptureAudio.cpp \
    Capture/CaptureDevices.cpp \
    Capture/CaptureDShow.cpp \
    Capture/CaptureImageFile.cpp \
    Capture/CaptureThread.cpp \
    Capture/CaptureXimea.cpp \
    Capture/VideoCapture.cpp \
    Common/BaseFfmpeg.cpp \
    Common/common.cpp \
    Common/GlobalAnimSettings.cpp \
    Common/PanoLog.cpp \
    Common/PanoQueue.cpp \
    Common/QTHandler.cpp \
    Common/SharedImageBuffer.cpp \
    Common/SlotInfo.cpp \
    Common/StreamFfmpeg.cpp \
    Common/TLogger.cpp \
    Common/TSystemSpec.cpp \
    Graphics/3DMath.cpp \
    Graphics/GLSLAntialiasing.cpp \
    Graphics/GLSLBlending.cpp \
    Graphics/GLSLBoundary.cpp \
    Graphics/GLSLBoxBlur.cpp \
    Graphics/GLSLColorCvt.cpp \
    Graphics/GLSLComposite.cpp \
    Graphics/GLSLFeathering.cpp \
    Graphics/GLSLFinalPanorama.cpp \
    Graphics/GLSLGainCompensation.cpp \
    Graphics/GLSLGaussianBlur.cpp \
    Graphics/GLSLLaplacian.cpp \
    Graphics/GLSLLiveViewer.cpp \
    Graphics/GLSLMultibandBlending.cpp \
    Graphics/GLSLPanoramaInteract.cpp \
    Graphics/GLSLPanoramaPlacement.cpp \
    Graphics/GLSLProgram.cpp \
    Graphics/GLSLPyramidSum.cpp \
    Graphics/GLSLResize.cpp \
    Graphics/GLSLSeam.cpp \
    Graphics/GLSLUnwarp.cpp \
    Graphics/GLSLWeightMap.cpp \
    Graphics/GLSLWeightVisualizer.cpp \
    Process/CalibProcess.cpp \
    Process/D360Parser.cpp \
    Process/D360Process.cpp \
    Process/D360Stitcher.cpp \
    Process/OculusViewer.cpp \
    Process/PtsParser.cpp \
    Process/StreamProcess.cpp \
    View/MCQmlCameraView.cpp \
    View/QmlInteractiveView.cpp \
    View/QmlMainWindow.cpp \
    View/QmlRecentDialog.cpp \
    Capture/CapturePtGrey.cpp

RESOURCES += qml.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# Default rules for deployment.
include(deployment.pri)

DISTFILES += \
    main.qml \
    qml/Header.qml \
    qml/Toolbox.qml \
    qml/Recent.qml \
    qml/RecentGriditem.qml \
    qml/RecentListitem.qml \
    qml/Status.qml \
    qml/components/Aboutbox.qml \
    qml/components/Anaglyphbox.qml \
    qml/components/Exposurebox.qml \
    qml/components/Helpbox.qml \
    qml/components/Interactivebox.qml \
    qml/components/InteractiveView.qml \
    qml/components/Livebox.qml \
    qml/components/Savebox.qml \
    qml/components/Settingsbox.qml \
    qml/components/Snapshotbox.qml \
    qml/components/Sphericalbox.qml \
    qml/components/Topbottombox.qml \
    qml/components/LiveSettingbox.qml \
    qml/components/DshowSettingbox.qml \
    qml/components/DecklinkSettingbox.qml \
    qml/components/LiveAudioitem.qml \
    qml/components/LiveCamListitem.qml \
    qml/components/VideoSettingbox.qml \
    qml/components/VideoListitem.qml \
    qml/components/VideoEmptyitem.qml \
    qml/components/ImageSettingbox.qml \
    qml/components/ImageListitem.qml \
    qml/components/Spliter.qml \
    qml/components/FloatingWindow.qml \
    qml/components/FloatingStitchWindow.qml \
    qml/components/MCCameraView.qml \
    qml/components/MCStitchCameraView.qml \
    qml/components/SphericalView.qml \
    qml/components/LiveView.qml \
    qml/components/MCVideoWindow.qml \
    qml/components/MCStitchVideoWindow.qml \
    qml/components/MdiArea.qml \
    qml/components/ToolWindowPositions.js \
    qml/components/ExposureDialog.qml \
    qml/components/ToolWindow.qml \
    qml/components/FloatingInteractiveWindow.qml \
    qml/components/MCInteractCameraView.qml \
    qml/components/MCInteractVideoWindow.qml \
    qml/components/CameraPreview.qml \
    qml/components/CameraSettingbox.qml \
    qml/components/VideoGlobalSettings.qml \
    qml/components/LiveTempCamSetting.qml \
    qml/components/ImageTempCamSetting.qml \
    qml/components/GroupCtrl.qml \
    qml/components/ControlPoint.qml \
    qml/components/SeamLabel.qml \
    qml/components/Notification.qml \
    qml/components/NotificationHistory.qml \
    qml/components/NotificationItem.qml \
    qml/components/ExposureCtrl.qml \
    qml/components/Line.qml \
    qml/components/BannerCtrl.qml \
    qml/components/CameraParams.qml \
    qml/components/OculusCtrl.qml \
    qml/components/OculusSwitch.qml \
    qml/components/BlendSettings.qml \
    qml/components/ScreenNumber.qml \
    qml/components/WeightmapWindow.qml



HEADERS += \
    Capture/AudioThread.h \
    Capture/BannerThread.h \
    Capture/CameraView.h \
    Capture/Capture.h \
    Capture/CaptureAudio.h \
    Capture/CaptureDevices.h \
    Capture/CaptureDShow.h \
    Capture/CaptureImageFile.h \
    Capture/CaptureThread.h \
    Capture/CaptureXimea.h \
    Capture/VideoCapture.h \
    Common/BaseFfmpeg.hpp \
    Common/common.h \
    Common/FfmpegUtils.hpp \
    Common/GlobalAnimSettings.h \
    Common/PanoLog.h \
    Common/PanoQueue.h \
    Common/QTHandler.h \
    Common/SharedImageBuffer.h \
    Common/SlotInfo.h \
    Common/StreamFfmpeg.hpp \
    Common/TLogger.h \
    Common/TSystemTypes.h \
    Graphics/3DMath.h \
    Graphics/GLSLAntialiasing.h \
    Graphics/GLSLBlending.h \
    Graphics/GLSLBoundary.h \
    Graphics/GLSLBoxBlur.h \
    Graphics/GLSLColorCvt.h \
    Graphics/GLSLComposite.h \
    Graphics/GLSLFeathering.h \
    Graphics/GLSLFinalPanorama.h \
    Graphics/GLSLGainCompensation.h \
    Graphics/GLSLGaussianBlur.h \
    Graphics/GLSLLaplacian.h \
    Graphics/GLSLLiveViewer.h \
    Graphics/GLSLMultibandBlending.h \
    Graphics/GLSLPanoramaInteract.h \
    Graphics/GLSLPanoramaPlacement.h \
    Graphics/GLSLProgram.h \
    Graphics/GLSLPyramidSum.h \
    Graphics/GLSLResize.h \
    Graphics/GLSLSeam.h \
    Graphics/GLSLUnwarp.h \
    Graphics/GLSLWeightMap.h \
    Graphics/GLSLWeightVisualizer.h \
    Include/BaseFfmpeg.hpp \
    Include/Buffer.h \
    Include/CaptureProp.h \
    Include/Config.h \
    Include/cTDate.h \
    Include/define.h \
    Include/ImageBuffer.h \
    Include/ImageHandler.h \
    Include/pts.h \
    Include/qedit.h \
    Include/RingBuffer.h \
    Include/Structures.h \
    Process/CalibProcess.h \
    Process/D360Parser.h \
    Process/D360Process.h \
    Process/D360Stitcher.h \
    Process/OculusViewer.h \
    Process/PtsParser.h \
    Process/StreamProcess.h \
    View/MCQmlCameraView.h \
    View/QmlInteractiveView.h \
    View/QmlMainWindow.h \
    View/QmlRecentDialog.h \
    Capture/CapturePtGrey.h


