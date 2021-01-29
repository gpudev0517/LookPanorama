import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Dialogs 1.2
import QtQuick.Extras 1.4
import Qt.labs.settings 1.0
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import "qml"
import "qml/components"
import "qml/controls"
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.2
import MCQmlCameraView 1.0
import QmlInteractiveView 1.0
import QmlMainWindow 1.0
import QmlRecentDialog 1.0
import QmlTakeManagement 1.0
import QmlApplicationSetting 1.0
import Pixmap 1.0

import QtQuick.Dialogs 1.1

QtObject{

property var  controlWindow: ApplicationWindow {
    id: root
    width: 1280
    height: 720
    color: "#000000"
    visible: true
    flags: Qt.Window | Qt.FramelessWindowHint
    visibility: Window.Maximized
    property int recentsize: 0
    property bool m_bRecent: false
    property string title
    Component.onCompleted: {
        root.createRecentList();
    }

    property bool       isFullScreen: true
    property int        curMode: 1
	property int		panoMode: 1
    property string     fileName:""
    //Others
    property string     headerTitle: "LookVR"
    property bool       isWait: true
    property bool       isLoaded: true
    property bool       isOculusWait: true
    property bool       isTemplate: false
    property bool       isStartStreaming: true
    property int        streamingXPos: 0
    property int        streamingYPos: 0

    property var        msg

    property var		selectedIniPath
    property var		isRecentConfigurationQueued: false

    property bool       isHiddenToolbarItem: false    

    property bool       isOnlyUsedImageOnToolbarItem: false

    property string     strloadingMessage

    onWidthChanged: {
        applicationSettingWindow.x  = root.width - 320
        applicationSettingWindow.y  = 25

        streamingXPos = root.width/2 - streamingbox.width/2
        streamingYPos = root.height/2 - streamingbox.height/2

//        if (applicationSettingWindow.uiTheme == "VRSE")
//            centeringToolbarItems()

//        if (sphericalToolbarItemsWidth() > (root.width - 150)){
//            autoHideIcons()
//            if (applicationSettingWindow.uiTheme == "VRSE")
//                centeringToolbarItems()
//        } else if (sphericalToolbarItemsWidth() < root.width - 150){
//            showHiddenIcons()
//            if (applicationSettingWindow.uiTheme == "VRSE")
//                centeringToolbarItems()
//        }

        if (root.width - 300 < 1165) {
            autoHideIcons()

            isHiddenToolbarItem = true
            if (applicationSettingWindow.uiTheme == "VRSE")
                centeringToolbarItems()
        }
        else {
            showHiddenIcons()

            isHiddenToolbarItem = false
            if (applicationSettingWindow.uiTheme == "VRSE")
                centeringToolbarItems()
        }
    }

    onHeightChanged: {
        applicationSettingWindow.x  = root.width - 320
        applicationSettingWindow.y  = 25

        streamingXPos = root.width/2 - streamingbox.width/2
        streamingYPos = root.height/2 - streamingbox.height/2
    }

    QmlRecentDialog {
        id: recentDialog
    }

    QmlTakeManagement {
        id: takeManagement
    }

    QmlApplicationSetting {
        id: _applicationSetting
    }

    QmlMainWindow {
        id:qmlMainWindow
        recDialog : recentDialog
        takManagement : takeManagement
        applicationSetting: _applicationSetting

		onCurIndexChanged:
		{
			takeManagementSettingbox.setCurrentModelIndex(qmlMainWindow.curIndex);
		}

        onElapsedTimeChanged:
        {
			if(root.panoMode == 1)
	            statusItem.elapseText = qmlMainWindow.elapsedTime
			else
	            playStatusItem.elapseText = qmlMainWindow.elapsedTime
        }

        onFpsChanged:
        {
			if(root.panoMode == 1)
	            statusItem.fpsText = qmlMainWindow.fps
			else
	            playStatusItem.fpsText = qmlMainWindow.fps
        }

        onSliderValueChanged:
        {
	        playStatusItem.sliderValue = qmlMainWindow.sliderValue
        }

        onExitChanged:
        {
            root.close();
            root.deleteCameraViews();
            Qt.quit();
        }

        onStarted:
        {
            maskID.visible = false;
            root.isWait = true;
        }

        onLoadingMessageReceived:
        {
            // RTMP Streaming
            if (loadingMessage == "StreamingSuccessed") {
                root.isStartStreaming = true
                statusItem.setStreamingItem(false)
                centralItem.closeStreamingbox()
            } else if (loadingMessage == "StreamingFailed") {
                root.isStartStreaming = true
                statusItem.setStreamingItem(false)
            }

            //
            var temploadingMessage = loadingMessage
            root.strloadingMessage = temploadingMessage

            loadingTimer.restart()
        }

        onDisconnectCompleted:
        {
            maskID.visible = false;
            root.isWait = true;

            if (qmlMainWindow.isCloseProject())
            {
                root.onBackByPause(false);
                forwardItem.visible = false;
            }

            recent.previousConfigurationClosed();
            root.previousConfigurationClosed();
            imageSettingbox.previousConfigurationClosed();
            dshowSettingbox.previousConfigurationClosed();
            videoSettingbox.previousConfigurationClosed();
        }

        onError:
        {
            maskID.visible = false;
            root.isWait = true;
            recent.visible = true;
        }

        onCalibratingFinished:
        {
            if (calibrated == true)
                busyLabel.text = "Calibrating succeed!";
            else
                busyLabel.text = "Calibrating failed!";
            calibTimer.restart()
        }

        onNotify:
        {
            //root.msg = notifyMsg.split(":");
            var notifyType = notifyMsg.split(":")[0];
            var contentStr = notifyMsg.split(":")[1];


            if (!root.isWait && (notifyType === "Error"))
            {
                root.isWait = true;
                maskID.visible = false;
            }

            notification.state = "expanded";
            //root.showNotify();
            root.showNotifyMsg(notifyType,contentStr);

            if (contentStr.substring(0,4) !== "Take") return
            statusItem.updateTakeMgr();
        }

        onPlayFinished: {
            statusItem.isPlaying = false;
            statusItem.setStreamingItem(false)
            statusItem.setPauseMode();
            statusItem.stopRecordTake();
        }

        onWeightMapEditCameraIndexChanged: {
            sphericalView.updateCameraIndex(weightMapEditCameraIndex)
        }

        onWeightMapEditUndoStatusChanged: {
            sphericalView.setUndoStatus(weightMapEditUndoStatus)
        }
        onWeightMapEditRedoStatusChanged: {
            sphericalView.setRedoStatus(weightMapEditRedoStatus);
        }
        onWeightMapResetStatusChanged: {
            sphericalView.resetWeightMapSettingView();
        }

        onTemplatePacFileLoadFinished: {
            sphericalView.reloadStitchCameraTemplateParameters();
        }

        onStopSingleCaptureChanged: {
            liveView.setStopCaptureStatus();
        }

        onSingleCalibMessageChanged:{
            liveView.setSingleCalibMessage(singleCalibMessage);
        }

        onSingleCalibStatusChanged: {
            if (singleCalibStatus == 5) { // "calibration sucess"
                liveView.createCameraTemplateWindow();
                liveView.setCalibrateStatus(false)
            }

            if (singleCalibStatus == 2) { // "Press Calibrate"
                liveView.setSingleCalibCaptureItem()
            }
        }

        onStrengthRatioChanged: {
            liveView.setSingleCalibStrengthRatio(strengthRatio)
        }

        onStrengthDrawColorChanged: {
            liveView.setSingleCalibStrengthDrawColor(strengthDrawColor);
        }

        onGrayLutDataChanged: {
            sphericalView.loadLutData(lutData, 0);
        }
        onRedLutDataChanged: {
            sphericalView.loadLutData(lutData, 1);
        }
        onGreenLutDataChanged: {
            sphericalView.loadLutData(lutData, 2);
        }
        onBlueLutDataChanged: {
            sphericalView.loadLutData(lutData, 3);
        }

    }

    function setFocus() {
        keyPressMonitorItem.focus = true;
    }

	function initPlayback(index) {
        statusItem.isPlaying = false;
        statusItem.setPauseMode();
        statusItem.stopRecordTake();

		qmlMainWindow.initPlayback(index);
		playStatusItem.resetPlayback();

		sphericalView.updatePlaybackConfiguration();
	}

	function onChangePanoMode() {
		if(panoMode === 1)
		{
			panoMode = 2
			statusItem.visible = false;
			playStatusItem.visible = true;
		}
		else
		{
			panoMode = 1
			statusItem.visible = true;
			playStatusItem.visible = false;
			qmlMainWindow.disconnectPlayback();
			sphericalView.updateConfiguration();
		}
		toolbox.changeFromPanoMode();
		curMode = 2;
		toolbox.clearSelected();
	}

    Item {
        id: keyPressMonitorItem
        focus: true;
        Keys.onPressed: {
            if((event.key == Qt.Key_1) && (event.modifiers & Qt.ControlModifier))
            {
                toolbox.onLive();
            }
            else if((event.key == Qt.Key_2) && (event.modifiers & Qt.ControlModifier))
            {
                toolbox.onSpherical();
            }
            else if((event.key == Qt.Key_3) && (event.modifiers & Qt.ControlModifier))
            {
                toolbox.onInterative();
            }
            else if((event.key == Qt.Key_Q) && (event.modifiers & Qt.ControlModifier))
            {
                toolbox.closeProject();
            }
            else if((event.key == Qt.Key_Z) && (event.modifiers & Qt.ControlModifier) && toolbox.isWeightmap == true)
            {
                qmlMainWindow.weightmapUndo();
            }
            else if ((event.key == Qt.Key_Y) && (event.modifiers & Qt.ControlModifier) && toolbox.isWeightmap == true)
            {
                qmlMainWindow.weightmapRedo();
            }
            else if ((event.key == Qt.Key_W) && (event.modifiers & Qt.ControlModifier) && sphericalView.visible == true)
            {
                if (toolbox.isWeightmap == true)
                    root.onCloseWeightmap();
                else
                    root.onOpenWeightmap();
            }
            else if ((event.key == Qt.Key_P) && (event.modifiers & Qt.ControlModifier) && toolbox.isWeightmap == true)
            {
                sphericalView.reverseWeightMapViewModeSwitchStatus();
            }
            else if ((event.key == Qt.Key_F5))
            {
                recent.initFavorite_TemplateGridList();
            }
        }
    }

    Timer {
        id: resetTimer
        interval: 2000
        running: false
        repeat: false
        onTriggered:
        {
            if (notification.state === "expanded")
                notification.state = "collapsed";

        }
    }

    Timer {
        id: notifyTimer
        interval: 3000
        running: false
        repeat: false
        onTriggered:
        {
            if(notification.state === "expanded")
                notification.state = "collapsed";
        }
    }

    Timer {
        id: busyTimer
        interval: 100
        running: false
        repeat: false
        onTriggered: {
            root.openIniProject(fileOpenDialog.fileUrl.toString());
        }
    }

    Timer {
        id: calibTimer
        interval: 1000
        running: false
        repeat: false
        onTriggered: {
            maskID.visible = false;
            root.isLoaded = true;
        }
    }

    Timer {
        id: oculusTimer
        interval: 1000
        running: false
        repeat: false
        onTriggered: {
            maskID.visible = false;
            root.isOculusWait = true;
        }
    }

    // File dialog for selecting image file from local file system
    FileDialog {
       id: fileOpenDialog
       title: "Select configuration file"
       nameFilters: [ "L3D file (*.l3d)", "INI file (*.ini)","All files (*)" ]
       selectMultiple: false

       onSelectionAccepted: {
           root.isTemplate = false;
           root.openCameraViews();
           busyTimer.restart();
       }
    }

    Header {
        id: headerItem
        width: parent.width
        height: 32
        z: 5

    }
    Text{
        id: currentName
        y: (32 - height) / 2
        z: 5
        anchors.left: parent.left
        anchors.leftMargin: 65
        text: qsTr(root.headerTitle);
        color: "#ffffff"
        font.pointSize: 14
    }

    Item {
        id: backItem
        z: 5
        width: 50
        height: 31
        visible: false

        Rectangle {
            id: backHoveredRectangle
            y: 1
            z: 1
            anchors.fill: parent
            color:"#1f1f1f"
            visible: false
        }

        Image {
            id: backImage
            width: 50
            height: 31
            z: 1
            fillMode: Image.PreserveAspectFit
            source: "resources/btn_back.PNG"
        }

        MouseArea {
            id: backMouseArea
            anchors.fill: parent
            z: 2
            hoverEnabled: true
            onEntered: backHoveredRectangle.visible = true
            onExited: backHoveredRectangle.visible = false
            onClicked: root.onBackByPause(true);
        }
    }
    Item {
           id: forwardItem
           z: 6
           width: 50
           height: 31
           visible: false

           Rectangle {
               id: forwardHoveredRectangle
               y: 1
               z: 1
               //anchors.fill: parent
               width: parent.width
               height: parent.height
               color:"#1f1f1f"
               visible: false
           }

           Image {
               id: forwardImage
               width: 50
               height: 31
               z: 1
               fillMode: Image.PreserveAspectFit
               source: "resources/btn_forward.png"
           }

           MouseArea {
               id: forwardMouseArea
               anchors.fill: parent
               z: 2
               hoverEnabled: true
               onEntered: forwardHoveredRectangle.visible = true
               onExited: forwardHoveredRectangle.visible = false
               onClicked: root.onForward()
           }
       }

    Rectangle {
        id: topbarTools
        y: 32
        z: 2
        width: parent.width
        height: 48
        color: "#1f1f1f"

        ToolbarItem {
            id: calibrationCtrl
            anchors.right: snapshotCtrl.left
            imgUrl: "../../resources/ic_calibration.png"
            title: "Lens Calibration"
            visible: root.curMode === 1 ? true: false
            theme: applicationSettingWindow.uiTheme
            autoHide: root.isHiddenToolbarItem
            fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"

            textColor: applicationSettingWindow.uiTheme == "Default"? "white": "#d0e3ef"

            onClicked: {
                liveView.createCalibrationSettingbox();
                qmlMainWindow.startSingleCameraCalibration()
                liveView.appendCameraCombo();
                root.showSingleCamCalibToolBox();
            }
        }

        ToolbarItem {
            id: snapshotCtrl
            anchors.right: cameraTemplateCtrl.left
            imgUrl: "../../resources/snapshot.png"
            title: "Snapshot"
            theme: applicationSettingWindow.uiTheme
            autoHide: root.isHiddenToolbarItem
            fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


            visible: root.curMode === 1 ? true: false

            onClicked: {
                statusItem.setPauseMode();
                snapshotBox.getSnapshotDir();
                groupCtrl.state = "expanded";
                snapshotCtrl.visible = false;
                cameraTemplateCtrl.visible = false;
                calibrationCtrl.visible = false;
            }
        }

        ToolbarItem {
            id: cameraTemplateCtrl
            anchors.right: moreCtrl.left
            imgUrl: "../../resources/cameraTemplate.png"
            title: "Camera Template"
            theme: applicationSettingWindow.uiTheme
            autoHide: root.isHiddenToolbarItem
            fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


            visible: root.curMode === 1 ? true: false

            onClicked: {
                liveView.createCameraTemplateWindow();
            }
        }

        ListView {
            id: screenNumListview
            anchors.right: weightmapItem.left
            anchors.rightMargin: 15
            y: (parent.height - 20) / 2
            z: 2
            visible: root.curMode === 2 ? true: false
            width: 250
            height: 25
            layoutDirection: "LeftToRight"
            orientation: ListView.Horizontal

            model: ListModel {
                id: numberList
            }

            delegate: Item {
                x: 5
                width: 35
                height: 20
                Row {
                    width: 35
                    ScreenNumber {
                        screenNum: number
                        isSelected: selected
                    }
                }
            }
        }

//        Item{
//            id: recordItem
//            anchors.right: weightmapItem.left
//            width: 68
//            height: 48
//            visible: root.curMode === 2 ? true: false
//            Rectangle{
//                id: recordHoverRectangle
//                width: parent.width
//                height: parent.height
//                color: "#353535"
//                visible: false
//            }
//            MouseArea {
//                x: recordHoverRectangle.x
//                z: 2
//                width: recordHoverRectangle.width
//                height: recordHoverRectangle.height
//                hoverEnabled: true
//                onEntered: recordHoverRectangle.visible = true;
//                onExited: recordHoverRectangle.visible = false;
//                onClicked: {
//					qmlMainWindow.travelSessionRootPath()
//                    sphericalView.createTakeManagementSettingbox();
//                }
//            }
//            Image {
//                id: recordImage
//                z: 1
//                x: (parent.width - width) / 2
//                y: (parent.height - height) / 2
//                width: 23
//                height: 23
//                fillMode: Image.PreserveAspectFit
//                source: "../resources/ico_rec.png"
//            }
//        }

        ToolbarItem {
                    id: weightmapItem
                    anchors.right: bannerItem.left
                    imgUrl: "../../resources/ico_brush.png"
                    title: root.isOnlyUsedImageOnToolbarItem?"": "Paint weight"
                    theme: applicationSettingWindow.uiTheme
                    autoHide: root.isHiddenToolbarItem
                    fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                    visible: root.curMode === 2 ? true: false
                    enabled: !statusItem.isRecording

                    onClicked: {
                        root.onOpenWeightmap();
                    }
                }

        ToolbarItem {
                    id: bannerItem
                    anchors.right: panoSnapshotItem.left
                    imgUrl: "../../resources/ico_banner.png"
                    title: root.isOnlyUsedImageOnToolbarItem?"":  "Banner"
                    theme: applicationSettingWindow.uiTheme
                    autoHide: root.isHiddenToolbarItem
                    fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                    visible: root.curMode === 2 ? true: false
                    enabled: !statusItem.isRecording

                    onClicked: {
                        statusItem.setPauseMode();
                        toolbox.clearTopCtrl();
                        bannerCtrl.state = "expanded";
                        sphericalView.isBanner = true;
                    }
                }

        ToolbarItem {
                    id: panoSnapshotItem
                    anchors.right: exposureCtrl.left
                    imgUrl: "../../resources/snapshot.png"
                    title: root.isOnlyUsedImageOnToolbarItem?"":  "Snapshot"
                    theme: applicationSettingWindow.uiTheme
                    autoHide: root.isHiddenToolbarItem
                    fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                    visible: root.curMode === 2 ? true: false
                    enabled: !statusItem.isRecording

                    onClicked: {
                        statusItem.setPauseMode();
                        snapshotBox.getSnapshotDir();
                        toolbox.clearTopCtrl();
                        groupCtrl.state = "expanded";
                    }
                }

        ToolbarItem {
                    id: exposureCtrl
                    anchors.right: ptsAndpacCtrl.left
                    imgUrl: "../../resources/ico_exposure.png"
                    title: root.isOnlyUsedImageOnToolbarItem?"":  "Exposure"
                    theme: applicationSettingWindow.uiTheme
                    autoHide: root.isHiddenToolbarItem
                    fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                    visible: root.curMode === 2 ? true: false
                    enabled: !statusItem.isRecording

                    onClicked: {
                        statusItem.setPauseMode();
                        toolbox.clearTopCtrl();
                        qmlMainWindow.setTempCameraSettings();
                        sphericalBox.isExposure = true;
                        sphericalBox.showExposureSetting();
                        sphericalBox.state = "expanded";
                    }
                }

        ToolbarItem {
                    id: ptsAndpacCtrl
                    anchors.right: lutCtrl.left
                    imgUrl: ""
                    title: "Import camera information"
                    theme: applicationSettingWindow.uiTheme
                    autoHide: root.isHiddenToolbarItem
                    fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                    visible: root.curMode === 2 ? true: false
                    enabled: !statusItem.isRecording

                    textColor: applicationSettingWindow.uiTheme == "Default"? "white": "#d0e3ef"

                    onClicked: {
                        ptsAndpacFileDialog.open();
                    }

                    FileDialog{
                        id:ptsAndpacFileDialog
                        title: "Select configuration file"
                        nameFilters: [ "Calib File (*.pac *.pts)", "All files (*)" ]
                        selectMultiple: false

                        onSelectionAccepted: {
                            var fileName = fileUrl.toString().substring(8); // Remove "file:///" prefix
                            qmlMainWindow.reloadCameraCalibrationFile(fileName);
                        }
                    }
                }

        ToolbarItem {
                    id: lutCtrl
                    anchors.right: ctCtrl.left
                    imgUrl: "../../resources/LUT.png"
                    title: root.isOnlyUsedImageOnToolbarItem?"":  "LUT"
                    theme: applicationSettingWindow.uiTheme
                    autoHide: root.isHiddenToolbarItem
                    fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                    visible: root.curMode === 2 ? true: false
                    enabled: !statusItem.isRecording

                    onClicked: {
                        sphericalView.showLutCtrl();
                    }
                }

        ToolbarItem {
                    id: ctCtrl
                    anchors.right: stitchCameraTemplateCtrl.left
                    imgUrl: "../../resources/colortemperature.png"
                    title: root.isOnlyUsedImageOnToolbarItem?"":  "Color temperature"
                    theme: applicationSettingWindow.uiTheme
                    autoHide: root.isHiddenToolbarItem
                    fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                    visible: root.curMode === 2 ? true: false

                    onClicked: {
                        if (toolbox.isCTSetting)
                            sphericalView.closeCTSettingbox();
                        else
                            sphericalView.openCTSettingbox();
                    }
                }

        ToolbarItem {
                    id: stitchCameraTemplateCtrl
                    anchors.right: moreCtrl.left
                    imgUrl: "../../resources/ic_RigTemplate.png"
                    title: root.isOnlyUsedImageOnToolbarItem?"":  "Rig template"
                    theme: applicationSettingWindow.uiTheme
                    autoHide: root.isHiddenToolbarItem
                    fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                    visible: root.curMode === 2 ? true: false
                    enabled: !statusItem.isRecording
                    onClicked: {
                        if (toolbox.isStitchCameraTemplate)
                            sphericalView.onCloseStitchCameraTemplateSettingbox();
                        else
                            sphericalView.onOpenStitchCameraTemplateSettingbox();
                    }
                }

        OculusSwitch{
            id: oculusSwitch
            visible: false
            anchors.right: parent.right

            theme: applicationSettingWindow.uiTheme
        }

        ToolbarItem {
                    id: moreCtrl
                    anchors.right: parent.right
                    imgUrl: "../../resources/more_control.png"
                    title: ""
                    theme: applicationSettingWindow.uiTheme
                    autoHide: root.isHiddenToolbarItem
                    isUsedImageButton: true
                    customMargin: 0

                    visible: true
                    enabled: !statusItem.isRecording

                    onClicked: {
                        root.onPressedMainMore()
                    }
                }

        WeightmapTopCtrl {
            id: weightmapTopCtrl
            anchors.right: parent.right
            anchors.rightMargin: - width
            x: moreCtrl.x
            width: 96
            height: 48
            state: "collapsed"

            states: [
                State {
                    name: "expanded"
                    PropertyChanges {
                        target: weightmapTopCtrl
                        width: 0
                        visible: true
                    }
                },
                State {
                    name: "collapsed"
                    PropertyChanges {
                        target: weightmapTopCtrl
                        width: 96
                        visible: false
                    }
                }
            ]

            transitions: [
                Transition {
                    NumberAnimation { target: weightmapTopCtrl; property: "width"; duration: 300 }
                    NumberAnimation { target: weightmapTopCtrl; property: "opacity"; duration: 300 }
                    NumberAnimation { target: weightmapTopCtrl; property: "visible"; duration: 300 }
                }
            ]

        }

        GroupCtrl {
            id: groupCtrl
            anchors.right: parent.right
            anchors.rightMargin: - width
            x: moreCtrl.x
            width: 138
            height: 48
            state: "collapsed"

            states: [
                State {
                    name: "expanded"
                    PropertyChanges {
                        target: groupCtrl
                        width: 0
                        visible: true
                        isHover: true
                    }
                },
                State {
                    name: "collapsed"
                    PropertyChanges {
                        target: groupCtrl
                        width: 138
                        visible: false
                        isHover: false
                    }
                }
            ]

            transitions: [
                Transition {
                    NumberAnimation { target: groupCtrl; property: "width"; duration: 300 }
                    NumberAnimation { target: groupCtrl; property: "opacity"; duration: 300 }
                    NumberAnimation { target: groupCtrl; property: "visible"; duration: 300 }
                    NumberAnimation { target: groupCtrl; property: "isHover"; duration: 300 }
                }
            ]

        }

        OculusCtrl {
            id: oculusGroup
            anchors.right: parent.right
            anchors.rightMargin: - width
            x: moreCtrl.x
            width: 138
            height: 48
            state: "collapsed"

            states: [
                State {
                    name: "expanded"
                    PropertyChanges {
                        target: oculusGroup
                        width: 0
                        visible: true
                        isHover: true
                    }
                },
                State {
                    name: "collapsed"
                    PropertyChanges {
                        target: oculusGroup
                        width: 138
                        visible: false
                        isHover: false
                    }
                }
            ]

            transitions: [
                Transition {
                    NumberAnimation { target: oculusGroup; property: "width"; duration: 300 }
                    NumberAnimation { target: oculusGroup; property: "opacity"; duration: 300 }
                    NumberAnimation { target: oculusGroup; property: "visible"; duration: 300 }
                    NumberAnimation { target: oculusGroup; property: "isHover"; duration: 300 }
                }
            ]

        }

        ExposureCtrl {
            id: exposureGroup
            anchors.right: parent.right
            anchors.rightMargin: - width
            x: moreCtrl.x
            width: 138
            height: 48
            state: "collapsed"

            states: [
                State {
                    name: "expanded"
                    PropertyChanges {
                        target: exposureGroup
                        width: 0
                        visible: true
                        isHover: true
                    }
                },
                State {
                    name: "collapsed"
                    PropertyChanges {
                        target: exposureGroup
                        width: 138
                        visible: false
                        isHover: false
                    }
                }
            ]

            transitions: [
                Transition {
                    NumberAnimation { target: exposureGroup; property: "width"; duration: 300 }
                    NumberAnimation { target: exposureGroup; property: "opacity"; duration: 300 }
                    NumberAnimation { target: exposureGroup; property: "visible"; duration: 300 }
                    NumberAnimation { target: exposureGroup; property: "isHover"; duration: 300 }
                }
            ]

        }

        BannerCtrl {
            id: bannerCtrl
            anchors.right: parent.right
            anchors.rightMargin: - width
            x: moreCtrl.x
            width: 138
            height: 48
            state: "collapsed"

            states: [
                State {
                    name: "expanded"
                    PropertyChanges {
                        target: bannerCtrl
                        width: 0
                        visible: true
                        isHover: true
                    }
                },
                State {
                    name: "collapsed"
                    PropertyChanges {
                        target: bannerCtrl
                        width: 138
                        visible: false
                        isHover: false
                    }
                }
            ]

            transitions: [
                Transition {
                    NumberAnimation { target: bannerCtrl; property: "width"; duration: 300 }
                    NumberAnimation { target: bannerCtrl; property: "opacity"; duration: 300 }
                    NumberAnimation { target: bannerCtrl; property: "visible"; duration: 300 }
                    NumberAnimation { target: bannerCtrl; property: "isHover"; duration: 300 }
                }
            ]

        }
    }

    Notification {
        id: notification
        //anchors.top: headerItem.bottom
        anchors.bottom: statusItem.bottom
        width: 600
        height: 110
        z: 10
        state: "collapsed"
        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: notification
                    x: root.width - 600

                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: notification
                    x: root.width
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: notification; property: "x"; duration: 300 ;
                    easing.type: Easing.Linear;easing.overshoot: 5 }
            }
        ]
    }

    /*************************************central layout**********************************/
    Item {
        id: centralItem
        width: parent.width - 50
        height: parent.height - headerItem.height - topbarTools.height - statusItem.height
        x: 50
        y: headerItem.height + topbarTools.height
        z: 1

        TakeManagementSettingbox {
            id: takeManagementSettingbox
            z:  -1
            x: root.width
            visible:  false
        }

        Streaming {
            id: streamingbox
            z:  15
            x: root.width / 2
            y: root.height/ 2
            visible:  false
        }

        LiveView{
            id: liveView
            visible: true
            state: "show"

            states: [
                State {
                    name: "show"
                    PropertyChanges {
                        target: liveView
                        visible: true
                    }
                },
                State {
                    name: "hidden"
                    PropertyChanges {
                        target: liveView
                        visible: false
                    }
                }
            ]

        }

         SphericalView {
            id: sphericalView
            visible: false
            state: "hidden"
            states: [
                State {
                    name: "show"
                    PropertyChanges {
                        target: sphericalView
                        visible: true
                    }
                },
                State {
                    name: "hidden"
                    PropertyChanges {
                        target: sphericalView
                        visible: false
                    }
                }
            ]
        }

        InteractiveView {
            id: interactiveView
            visible: false
            state: "hidden"
            states: [
                State {
                    name: "show"
                    PropertyChanges {
                        target: interactiveView
                        visible: true
                    }
                },
                State {
                    name: "hidden"
                    PropertyChanges {
                        target: interactiveView
                        visible: false
                    }
                }
            ]
        }

        function createTakeManagementSettingbox(){
            takeManagementSettingbox.visible = true;
            takeManagementSettingbox.initTakeManagement();
        }

        function updateTakeManagementSettingbox(){
            takeManagementSettingbox.x = centralItem.width - takeManagementSettingbox.width
            takeManagementSettingbox.y = 0;
            takeManagementSettingbox.z = 10;
        }

        function createStreamingbox() {
            streamingbox.visible = true

            streamingbox.x = root.streamingXPos
            streamingbox.y = root.streamingYPos

            streamingbox.createResoltuionTypeModel()
            if (streamingbox.curWidth == -1) // first open
                streamingbox.initializeResolution(qmlMainWindow.getTempPanoWidth(), qmlMainWindow.getTempPanoHeight())

            if (qmlMainWindow.getStreamingPath() !== "")
                streamingbox.setStreamingPath(qmlMainWindow.getStreamingPath())
        }

        function closeStreamingbox() {
            streamingbox.visible = false
        }

        function initTakeManagementSettingbox(){
            takeManagementSettingbox.visible = false;
            updateTakeManagementSettingbox();
        }

        function getTakeManangementComment() {
            return takeManagementSettingbox.getTakeManangementComment();
        }

        function setTakeSessionPath(strTakeSessionPath) {
            takeManagementSettingbox.setCapturePath(strTakeSessionPath)
        }

    }

    /*************************************status layout**********************************/

    Status {
        id:statusItem
        y:parent.height - 105
        width: parent.width
        height: 105
        z: 2
    }

	PlayStatus {
        id:playStatusItem
        y:statusItem.y
        width: statusItem.width
        height: statusItem.height
        z: 2
    }

    /************************************************************************************/

    Recent {
        id: recent
        x: 0
        y: 32
        z: 3
        width: parent.width
        height: parent.height - 32

        state: "expanded"

        states: [
            State {
                name: "collapsed"
                PropertyChanges {
                    target: recent
                    visible: false
                }
            },
            State {
                name: "expanded"
                PropertyChanges {
                    target: recent
                    visible: true
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: recent; property: "width"; duration: 100 }
                NumberAnimation { target: recent; property: "opacity"; duration: 100 }
            }
        ]
    }


    Toolbox {
        id: toolbox
        height: root.height - 32
        x: 0
        y: 32
        z: 2

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: toolbox
                    width: 250
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: toolbox
                    width: 50
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: toolbox; property: "width"; duration: 300}
                NumberAnimation { target: toolbox; property: "opacity"; duration: 300 }
            }
        ]
    }

    Livebox {
        id: liveBox
        x: parent.width - width
        y: 32
        z: 2

        height: root.height - y

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: liveBox
                    width: 350
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: liveBox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: liveBox; property: "width"; duration: 100 }
                NumberAnimation { target: liveBox; property: "opacity"; duration: 100 }
            }
        ]
    }

    Sphericalbox {
        id: sphericalBox
        x: parent.width - width
        y: 32
        z: 3

        height: root.height - y

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: sphericalBox
                    width: 350
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: sphericalBox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: sphericalBox; property: "width"; duration: 100 }
                NumberAnimation { target: sphericalBox; property: "opacity"; duration: 100 }
            }
        ]
    }

    Interactivebox {
        id: interactiveBox
        x: parent.width - width
        y: 32
        z: 4

        height: root.height - y

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: interactiveBox
                    width: 350
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: interactiveBox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: interactiveBox; property: "width"; duration: 100 }
                NumberAnimation { target: interactiveBox; property: "opacity"; duration: 100 }
            }
        ]
    }

    ApplicationSetting {
        id: applicationSettingWindow
        x: 0
        y: 0
        z: 14
        visible: false
        width: 220
        height: 250

        onUiThemeChanged: {
            if (uiTheme == "Default") {
                cameraTemplateCtrl.anchors.rightMargin = 0
                stitchCameraTemplateCtrl.anchors.rightMargin = 0
                oculusSwitch.anchors.rightMargin = 0
            } else if (uiTheme == "VRSE") {
                root.centeringToolbarItems()
            }
        }
    }


    Snapshotbox {
        id: snapshotBox
        x: parent.width - width
        y: 32
        z: 5

        height: root.height - y

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: snapshotBox
                    width: 350
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: snapshotBox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: snapshotBox; property: "width"; duration: 100 }
                NumberAnimation { target: snapshotBox; property: "opacity"; duration: 100 }
            }
        ]
    }

    /*Exposurebox {
        id: exposureBox
        x: parent.width - width
        y: 32
        z: 6

        height: root.height - y

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: exposureBox
                    width: 350
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: exposureBox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: exposureBox; property: "width"; duration: 100 }
                NumberAnimation { target: exposureBox; property: "opacity"; duration: 100 }
            }
        ]
    }

    Topbottombox {
        id: topBottomBox
        x: parent.width - width
        y: 32
        z: 7

        height: root.height - y

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: topBottomBox
                    width: 250
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: topBottomBox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: topBottomBox; property: "width"; duration: 100 }
                NumberAnimation { target: topBottomBox; property: "opacity"; duration: 100 }
            }
        ]
    }

    Anaglyphbox {
        id: anaglyphBox
        x: parent.width - width
        y: 32
        z: 8

        height: root.height - y

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: anaglyphBox
                    width: 250
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: anaglyphBox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: anaglyphBox; property: "width"; duration: 100 }
                NumberAnimation { target: anaglyphBox; property: "opacity"; duration: 100 }
            }
        ]
    }*/

    NotificationHistory {
        id: notifyHistorybox
        x: parent.width - width
        y: 32
        z: 2

        height: root.height - y

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: notifyHistorybox
                    //width: 350
                    x: root.width - 350
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: notifyHistorybox
                    //width: 0
                    x: root.width
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: notifyHistorybox; property: "x"; duration: 100 }
                NumberAnimation { target: notifyHistorybox; property: "opacity"; duration: 100 }
            }
        ]
    }

    Savebox {
        id: saveBox
        x: parent.width - width
        y: 32
        z: 9

        height: root.height - y

        state: "collapsed"

        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: saveBox
                    width: 350
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: saveBox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: saveBox; property: "width"; duration: 100 }
                NumberAnimation { target: saveBox; property: "opacity"; duration: 100 }
            }
        ]
    }

    VideoSettingbox {
        id: videoSettingbox
        x: parent.width - width
        y: headerItem.height
        z: 3
        height: parent.height
        state: "collapsed"
        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: videoSettingbox
                    width: root.width
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: videoSettingbox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: videoSettingbox; property: "width"; duration: 100 }
                NumberAnimation { target: videoSettingbox; property: "opacity"; duration: 100 }
            }
        ]
    }

    ImageSettingbox {
        id: imageSettingbox
        x: parent.width - width
        y: headerItem.height
        z: 2
        height: parent.height
        state: "collapsed"
        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: imageSettingbox
                    width: root.width
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: imageSettingbox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: imageSettingbox; property: "width"; duration: 100 }
                NumberAnimation { target: imageSettingbox; property: "opacity"; duration: 100 }
            }
        ]
    }

    DshowSettingbox {
        id: dshowSettingbox
        x: parent.width - width
        y: headerItem.height
        z: 2
        height: parent.height
        state: "collapsed"
        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: dshowSettingbox
                    width: root.width
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: dshowSettingbox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: dshowSettingbox; property: "width"; duration: 100 }
                NumberAnimation { target: dshowSettingbox; property: "opacity"; duration: 100 }
            }
        ]
    }

    Rectangle{
        id: maskID
        anchors.fill: parent
        y:headerItem.height
        height:parent.height - headerItem.height
        width: parent.width
        visible: false
        color: "#171717"
        opacity: 0.7
        z:12
        MouseArea {
           id: mouseArea1
           anchors.fill: parent
           hoverEnabled: true
        }

        Text {
           id: busyLabel
           z: 13
           y: parent.height / 2 + 30
           text: "Starting..."
           width: Math.min(maskID.width, maskID.height) * 0.8
           height: font.pixelSize
           anchors.horizontalCenter: parent.horizontalCenter
           renderType: Text.NativeRendering//Text.QtRendering
           font.pixelSize: 30
           horizontalAlignment: Text.AlignHCenter
           fontSizeMode: Text.Fit
           font.family: Flat.FlatStyle.fontFamily
           font.weight: Font.Light
           color: "#ffffff"
       }
    }


    WorkerScript {
        id: worker

        source: "./qml/components/Workerscript.js"

        onMessage: {
            var msg = messageObject.result
            root.setloadingMessage(msg)
        }
    }

    Timer {
        id: loadingTimer
        interval: 1
        running: true
        repeat: true

        onTriggered: {
            worker.sendMessage({'temp': root.strloadingMessage})
        }
    }

    function setloadingMessage(loadingMessage) {
        busyLabel.text = loadingMessage
        loadingTimer.stop()
    }

    BusyIndicator {
        running: !root.isWait
        anchors.centerIn: parent
        z:13
    }

    BusyIndicator {
        running: !root.isLoaded
        anchors.centerIn: parent
        z: 13
    }

    BusyIndicator {
        running: !root.isOculusWait
        anchors.centerIn: parent
        z: 13
    }

    BusyIndicator {
        running: !root.isStartStreaming
        anchors.centerIn:  parent
        z: 13
    }

//    function showNotify()
//    {
//        if (root.msg[0] === "Error")
//        {
//            notification.imagePath = "../../resources/ico_error.png"
//            notification.typeText = "Error"
//        }
//        else if(root.msg[0] === "Warning")
//        {
//            notification.imagePath = "../../resources/ico_warning.png"
//            notification.typeText = "Warning"
//        }
//        else if(root.msg[0] === "Information")
//        {
//            notification.imagePath = "../../resources/ico_notify.png"
//            notification.typeText = "Information"
//        }

//        notification.contentText = root.msg[1];
//        notifyTimer.restart();
//    }

    function showNotifyMsg(type,contentStr) {
        if (type === "Error") {
            notification.imagePath = "../../resources/ico_error.png";
            notification.typeText = "Error";
        } else if (type === "Warning") {
            notification.imagePath = "../../resources/ico_warning.png"
            notification.typeText = "Warning";
        } else if (type === "Information") {
            notification.imagePath = "../../resources/ico_notify.png";
            notification.typeText = "Information";
        }
        notification.contentText = contentStr;
        notification.state = "expanded";
        notifyTimer.restart();

    }

    function openCameraViews() {
        maskID.visible = true
        busyLabel.text = "Starting...";
        root.isWait = false
    }

    function onCalibrate()
    {
        maskID.visible = true;
        busyLabel.text = "Calibrating...";
        root.isLoaded = false;
    }

    function onOculusWait()
    {
        maskID.visible = true;
        busyLabel.text = "Wait for a moment...";
        root.isOculusWait = false;
    }

    function onPressedMore() {
        toolbox.showSpecifyBox()
    }
    function onPressedMainMore() {
        toolbox.showMainBox()

    }

    function onBackByPause(isNeedPause) {
        if (isNeedPause)
            statusItem.setPauseMode();
        toolbox.clearSelected();
        backItem.visible = false;
        forwardItem.visible = true;
        recent.state = "expanded";
        sphericalWindow.clearSphericalWindow();
        liveBox.initControlPoint();		

		recent.initFavorite_TemplateGridList();
    }

    function onForward() {
        toolbox.clearSelected()
        statusItem.setPlayAndPause();
        recent.state = "collapsed"
        backItem.visible = true
        forwardItem.visible = false
    }

    function showBack()
    {
        backItem.visible = true;
        forwardItem.visible = false;
    }

    function clearBackForward()
    {
        backItem.visible = false;
        forwardItem.visible = false;
    }

    function onCloseCTSettingbox() {
        sphericalView.closeCTSettingbox();
    }

    function onCloseLutCtrl() {
    	sphericalView.closeLutCtrl();
    }

    function onCloseStitchCameraTemplateSettingbox() {
        sphericalView.onCloseStitchCameraTemplateSettingbox();
    }

    function onCloseWeightmap() {
        weightmapTopCtrl.state = "collapsed";
        toolbox.isWeightmap = false;
        toolbox.initSphericalTopControls();
        sphericalView.closeWeightmapSettingbox();
        qmlMainWindow.setWeightMapEditMode(false);
    }

    function onOpenWeightmap() {
        sphericalView.isWeightMap = true
        qmlMainWindow.setTempCameraSettings();
        toolbox.isWeightmap = true
        statusItem.setPauseMode();
        toolbox.state = "collapsed";
        qmlMainWindow.setWeightMapEditMode(true);
        sphericalView.createWeightmapSettingbox();
        toolbox.clearTopCtrl();
        weightmapTopCtrl.state = "expanded";

        onCloseCTSettingbox();
        onCloseStitchCameraTemplateSettingbox();
    }

    function onFileOpen()
    {
        fileOpenDialog.open()
    }

    function createRecentList()
    {
        root.m_bRecent = qmlMainWindow.openRecentMgrToINI()
        if(!root.m_bRecent)
        {
            console.log("Loading recent file list failed!");
            return;
        }

        for (var i = 0; i < qmlMainWindow.getRecentCnt(); i++) {
            var title = qmlMainWindow.getRecentTitle(i);
            var fullPath = qmlMainWindow.getRecentFullPath(i);
            var shortPath = qmlMainWindow.getRecentPath(i);
            var type = qmlMainWindow.getRecentType(i);

            recent.append(title, fullPath, shortPath, type);
        }
    }
    function setCurrentTitle(fileName){
        var projectName = "LookVR";
        if (fileName == "")
            currentName.text = projectName;
        else
            currentName.text = fileName + " - " + projectName;
    }

    function openIniProject(strIniPath)
    {
        selectedIniPath = strIniPath.toString().substring(8); // Remove "file:///" prefix;

        //root.openCameraViews();

        if (qmlMainWindow.isEmpty)
        {
            onOpenProject();
        }
        else
        {
            isRecentConfigurationQueued = true;
            toolbox.closeProject();
        }
    }

    function previousConfigurationClosed()
    {
        if (isRecentConfigurationQueued)
        {
            isRecentConfigurationQueued = false;
            onOpenProject();
        }
    }

    function onOpenProject()
    {
        var recentIndex = qmlMainWindow.openIniPath(selectedIniPath);
        addRecentList(recentIndex);

        initVariants();
        initUI();

        qmlMainWindow.openProject();
        createCameraViews();
        createCalibCameraView();

        setCurrentTitle(qmlMainWindow.getRecentTitle(0) + ".l3d");

        toolbox.clearSelected();
		
	   root.panoMode = 2;
       root.onChangePanoMode();
    }

    function initUI() {
        recent.state = "collapsed";
        backItem.visible = true;
        forwardItem.visible = false;
        statusItem.initPlayAndRecord();
        toolbox.initLiveTopControls();
        sphericalWindow.visible = false;
        sphericalView.isBanner = false;
        bannerCtrl.clearAllBannerList();
        liveView.initDetailsWindow();
        liveView.destroyCameraTemplateWindow();
        sphericalView.initWeightmapSettingbox();
        centralItem.initTakeManagementSettingbox();
        createNumberList();
        statusItem.elapseText = qsTr("00:00:00.000");
    }

    function deleteCameraViews() {
        liveView.clearLiveCamView();
    }

    function initVariants() {
        toolbox.hoverType = 2;
        root.curMode = 1;
		root.panoMode = 1;
		statusItem.initPlayAndRecord()
		toolbox.isWeightmap = false;
        toolbox.isSetting = false;
        sphericalWindow.isCreated = false;
        liveBox.isDetailed = false;
        sphericalView.isSeam = false;
    }

    function addRecentList(recentIndex) {
        if (recentIndex < 0) return;	 // If loading ini file error
        if (recent.getCount() < qmlMainWindow.getRecentCnt()) { // Not new ini
            recent.addTop(recentIndex, false);
        } else {
            recent.addTop(recentIndex, true);
        }

        qmlMainWindow.saveRecentMgrToINI();
    }

	function isRigTemplateMode() {
		return !(qmlMainWindow.getTemplateIniFileName().length == 0);
	}

	function getSelectedRigTemplateFile() {
		return qmlMainWindow.getTemplateIniFileName();
	}

    function createCameraViews()
    {
        console.log("Create views ...");

        liveView.initializeLiveView();

        sphericalView.visible = false;
        sphericalView.updateConfiguration();
        sphericalWindow.clearSphericalWindow();

        interactiveView.visible = false;
        interactiveView.updateConfiguration();

        console.log("Done creating views ...");
    }

    function showSingleCamCalibToolBox() {

        liveView.updateCalibrationSettingsbox()
        sphericalView.visible = false;
        sphericalView.updateConfiguration();
        sphericalWindow.clearSphericalWindow();
        interactiveView.visible = false;
        interactiveView.updateConfiguration();
    }

    function createCalibCameraView()
    {
        liveView.updateCalibrationSettingsbox()
    }

    function createNumberTab() {
        var numCount = qmlMainWindow.getScreenCount();
        for(var i = 0; i < numCount; i++) {
            screenNumber.addTab(i + 1);
        }
        screenNumber.width = 35 * numCount;
     }

    function createNumberList()
    {
        numberList.clear();

        var numCount = qmlMainWindow.getScreenCount();
        for(var i = 0; i < numCount; i++)
        {
            numberList.append({"number":  (i + 1),"selected": false})
        }

        screenNumListview.width = numCount * 35;
    }

    function closeCalibSettingBox()
    {
        liveView.closeCalibSettignBox();
    }

    function closeWeightMapSettingsBox()
    {
        sphericalView.closeWeightmapSettingbox();
        sphericalView.isWeightMap = false
    }

    function centeringToolbarItems() {
        var groupItemsWidth = calibrationCtrl.width + snapshotCtrl.width + cameraTemplateCtrl.width
        var rightMargin = root.width/2 - moreCtrl.width - groupItemsWidth/2
        cameraTemplateCtrl.anchors.rightMargin = rightMargin

        groupItemsWidth = stitchCameraTemplateCtrl.width + ctCtrl.width + lutCtrl.width + ptsAndpacCtrl.width + exposureCtrl.width + panoSnapshotItem.width + bannerCtrl.width + weightmapItem.width + screenNumListview.width
        rightMargin = root.width/2 - moreCtrl.width - groupItemsWidth/2
        stitchCameraTemplateCtrl.anchors.rightMargin = rightMargin

        groupItemsWidth = oculusSwitch.width
        rightMargin = root.width/2 - moreCtrl.width - groupItemsWidth/2
        oculusSwitch.anchors.rightMargin = rightMargin
    }

    function autoHideIcons() {
        isOnlyUsedImageOnToolbarItem = true
        
    }

    function showHiddenIcons() {
        isOnlyUsedImageOnToolbarItem = false
       
    }

    function sphericalToolbarItemsWidth() {
        var width = stitchCameraTemplateCtrl.width + ctCtrl.width + lutCtrl.width + ptsAndpacCtrl.width + exposureCtrl.width + panoSnapshotItem.width + bannerCtrl.width + weightmapItem.width + screenNumListview.width
        return width
    }
}
  property var sphericalWindow: ApplicationWindow {
        id: sphericalWindow
        width: centralItem.width
        height: centralItem.height
        color: "#000000"
        title: "SphericalView"
        flags: Qt.Window | Qt.FramelessWindowHint
        visible: false

        property bool isCreated: false

        MCStitchCameraView {
            id: fullscreenObject
            x: 0
            y: 0
            floatingWindowBorderColor: "#00000000"
        }

        Item {
            id: keyItem
            focus: true;
            Keys.onPressed: {
                if(event.key ===Qt.Key_Escape) {
                   sphericalWindow.clearSphericalWindow()
                }
            }
        }

        function createSphericalWindow(screenNum) {
            sphericalWindow.visible = true;
            if(sphericalWindow.isCreated) return;
            sphericalWindow.isCreated = true;
            sphericalView.updateFullscreen(screenNum);
        }

        function clearSphericalWindow() {
            sphericalWindow.visible = false;
            root.createNumberList();
        }
    }
}
