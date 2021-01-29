import QtQuick 2.5
import QtQuick.Dialogs 1.2
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4

import Look3DQmlModule 1.0
import "components"

Item {
    id: toolBox
    width : 250
    height: 684
    opacity: 1

    property bool isHovered : false
    property bool isSelected: false
    property int hoverType:0
    property int heightVal: 50
    property int clickCnt: 0
    property bool isWeightmap: false
    property bool isSetting: false
	property bool isCTSetting: false
	property bool isStitchCameraTemplate: false

    Component.onCompleted: {
    }

    Text {
        id: titleText
        x: 70
        y: 13
        color: "#ffffff"
        text: qsTr("")
        z: 3
        font.bold: false
        font.pixelSize: 20
    }

    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#171717"
        opacity: 0.9
    }

    Image {
        id: groupToolBoxImage
        x: 0
        y: 0
        width: 50
        height: 684
        z: 1
        fillMode: Image.PreserveAspectFit
        source: "../resources/btn_group_toolbox.png"

        MouseArea {
            id: listMouseArea
            width: parent.width
            height: 50
            enabled: true
            hoverEnabled: true
            onHoveredChanged: {
                hoverType = 1
                onHovered()
            }
            onClicked:toolbox.state == "expanded" ? toolbox.state = "collapsed" : toolbox.state = "expanded"
        }

        MouseArea {
            id: liveMouseArea
            x: 0
            y: 50
            enabled: true
            height: 50
            width: toolbox.width
            hoverEnabled: true
            onHoveredChanged: {
                hoverType = 2
                onHovered()
            }
            onClicked:{
                onLive()
            }
        }

        MouseArea {
            id: sphericalMouseArea
            x: 0
            y: 100
            enabled: true
            width: toolbox.width
            height: 50
            hoverEnabled: true
            onHoveredChanged: {
                hoverType = 3
                onHovered()
            }
            onClicked: {
                onSpherical()
            }
        }

        MouseArea {
            id: interativeMouseArea
            x: 1
            y: 150
            enabled: true
            width: toolbox.width
            height: 50
            hoverEnabled: true
            onHoveredChanged: {
                hoverType = 4
                onHovered()
            }
            onClicked: {
                onInterative()
            }
        }


		MouseArea {
            id: closeProjectMouseArea
            x: 3
            y: 200
            enabled: true
            width: toolbox.width
            height: 50
            hoverEnabled: true
            onHoveredChanged: {
                hoverType = 5;
                onHovered();
            }
            onClicked: {
                closeProject();
            }
        }

    }

	Image {
        id:  closeImage
        x: 20
        y: 216
        width: 15
        height: 15
        visible: true
        z: 1
        source: "../resources/uncheck.png"
        fillMode: Image.PreserveAspectFit
    }

    Spliter{
        id: bottomSpliter
        anchors.bottom: notifyHistoryItem.top
        width: parent.width
        z: 3
    }



    Image {
        id: saveImage
        source: "../resources/save.png"
        anchors.bottom: saveAsImage.top
        anchors.bottomMargin: 10
        width: 50
        height: 50
        fillMode: Image.PreserveAspectFit
        z: 1

        MouseArea {
            id: saveMouseArea
            width: toolbox.width
            height: 50
            enabled: !isWeightmap
            hoverEnabled: true
            onHoveredChanged: {
                hoverType = 14;
                onHovered();
            }
            onClicked: {
                onSave();
            }
        }

    }

    Image {
        id: saveAsImage
        source: "../resources/save_as.png"
        anchors.bottom: reloadImage.top
        anchors.bottomMargin: 18
        x: 14
        width: 25
        height: 25
        fillMode: Image.PreserveAspectFit
        z: 1

        MouseArea {
            id: saveAsMouseArea
            width: toolbox.width
            height: 50
            enabled: true
            hoverEnabled: true
            onHoveredChanged: {
                hoverType = 13;
                onHovered();
            }
            onClicked: {
                onSaveAs();

            }
        }

    }

	Image {
        id: reloadImage
        source: "../resources/icon_reload.png"
        anchors.bottom: bottomGroupToolBoxImage.top
        width: 50
        height: 50
        fillMode: Image.PreserveAspectFit
        z: 1

        MouseArea {
            id: reloadImageMouseArea
            width: toolbox.width
            height: 50
            enabled: true
            hoverEnabled: true
            onHoveredChanged: {
                hoverType = 12;
                onHovered();
            }
            onClicked: {
                onReloadConfig();
            }
        }

    }

    Image {
        id: bottomGroupToolBoxImage
        x: -1
        anchors.bottom: notifyHistoryItem.top
        width: 50
        height: 50
        fillMode: Image.PreserveAspectFit
        MouseArea {
            id: settingMouseArea
            width: toolbox.width
            height: 50
            enabled: true
            hoverEnabled: true
            onHoveredChanged:  {
                hoverType = 11
                onHovered()
            }
            onClicked: {
                onSettings()
            }
        }

        z: 1
        source: "../resources/setting.png"
    }

    Item {
        id: notifyHistoryItem
        width: 50
        height: 50
        anchors.bottom: parent.bottom
        z: 1

        Image {
            id: notifyHistoryImage
            source: "../resources/notifyHistory.png"
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            width: 25
            height: 25
            fillMode: Image.PreserveAspectFit
            z: 1
        }

        MouseArea {
            id: historyMouseArea
            width: toolbox.width
            height: 50
            enabled: true
            hoverEnabled: true
            onHoveredChanged: {
                hoverType = 10;
                onHovered();
            }
            onClicked: {
                onNotifyHistory();

            }
        }
    }

    Rectangle {
        id: hoveredRectangle
        x: 0
        y: -201
        width: parent.width
        height: 50
        color: "#1f1f1f"
        z: 0
        opacity: 0.9
    }

    Rectangle {
        id: selectedRectangle
        x: 0
        y: -145
        width: parent.width
        height: 50
        color: "#0e3e64"
        z: 0
        opacity: 0.9
    }

    Rectangle {
        id: disableRectangle
        x: 0
        y: -255
        width: parent.width
        height: 50
        color: "#1f1f1f"
        z: 1
        opacity: 1.0
    }

    ScrollView {
        id: scrollView
        width: parent.width
        height: parent.height
        opacity: 0.8
        verticalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        flickableItem.interactive: true

        style: ScrollViewStyle {
            transientScrollBars: false
            handle: Item {
                implicitWidth: 14
                implicitHeight: 26
                Rectangle {
                    color: "#424246"
                    anchors.fill: parent
                    anchors.topMargin: 6
                    anchors.leftMargin: 4
                    anchors.rightMargin: 4
                    anchors.bottomMargin: 6
                }
            }

            scrollBarBackground: Item {
                implicitWidth: 14
            }

            decrementControl: Rectangle{
                visible: false
            }

            incrementControl: Rectangle{
                visible: false
            }
            corner: Rectangle{
                visible:false
            }
        }


        Item{
            id: groupItems
            width: scrollView.width
            height: scrollView.height

            Item {
                id: titleGroup
                x: 68
                y: 1
                width: parent.width * 3 / 4
                height: parent.height
                z: 1
                rotation: 0
                Text {
                    id: liveText
                    x: 1
                    y: 67
                    color: "#8a8a8a"
                    text: qsTr("Live")
                    font.pixelSize: 15
                }

                Text {
                    id: sphericalText
                    x: 1
                    y: 114
                    color: "#8a8a8a"
                    text: qsTr("Spherical")
                    font.pixelSize: 15
                }

                Text {
                    id: interactiveText
                    x: 1
                    y: 164
                    color: "#8a8a8a"
                    text: qsTr("Interactive")
                    font.pixelSize: 15
                }

//                Text {
//                    id: pauseText
//                    x: 1
//                    y: 214
//                    color: "#8a8a8a"
//                    text: qsTr("Start")
//                    font.pixelSize: 15
//                }

//                Text {
//                    id: recordText
//                    x: 1
//                    y: 265
//                    color: "#8a8a8a"
//                    text: qsTr("Start Record")
//                    font.pixelSize: 15
//                }

				Text {
                    id: closeText
                    x: 1
                    y: 215
                    color: "#8a8a8a"
                    text: qsTr("Close")
                    font.pixelSize: 15
                }

                Text {
                    id: saveText
                    x: 1
                    anchors.bottom: saveAsText.top
                    anchors.bottomMargin: 30
                    color: "#8a8a8a"
                    text: qsTr("Save")
                    font.pixelSize: 15
                }

                Text {
                    id: saveAsText
                    x: 1
                    anchors.bottom: reloadText.top
                    anchors.bottomMargin: 33
                    color: "#8a8a8a"
                    text: qsTr("Save As")
                    font.pixelSize: 15
                }

				Text {
                    id: reloadText
                    x: 1
                    anchors.bottom: settingText.top
                    anchors.bottomMargin: 33
                    color: "#8a8a8a"
                    text: qsTr("Open")
                    font.pixelSize: 15
                }

                Text {
                    id: settingText
                    x: 0
                    anchors.bottom: notifyHistoryText.top
                    anchors.bottomMargin: 33
                    color: "#8a8a8a"
                    text: qsTr("Settings")
                    font.pixelSize: 15
                }

                Text {
                    id: notifyHistoryText
                    x: 1
//                    anchors.bottom: saveText.top
//                    anchors.bottomMargin: 33
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 17
                    color: "#8a8a8a"
                    text: qsTr("Notifications")
                    font.pixelSize: 15
                }


                Text {
                    id: currentTitleText
                    x: 0
                    y: 12
                    color: "#ffffff"
                    text: qsTr("")
                    font.bold: true
                    font.pixelSize: 19
                }
            }
        }
    }

    Timer {
        id: delaytimer
        interval: 100
        running: false
        repeat: false
        onTriggered: {
            if(toolbox.state == "collapsed" && !isSelected)
                toolbox.state = "expanded"
        }
    }
    function onHovered() {
        isHovered = !isHovered
        var text

        if(isHovered) {
            isSelected = false

            if(hoverType > 9) {
                hoveredRectangle.y = backgroundRectangle.height  - 50 * (hoverType - 9)
                text = titleGroup.childAt(3,backgroundRectangle.height  - 50 * (hoverType - 9) + 25)
            }
            else {
                text = titleGroup.childAt(3,17 + (hoverType - 1) * 50)
                hoveredRectangle.y = 50 * (hoverType - 1)
            }

            if(text)
                text.color = "#ffffff"

            delaytimer.restart()
        }else{
           // delaytimer.stop()

            clearHovered()

            if(hoverType > 9)
                text = titleGroup.childAt(3,backgroundRectangle.height  - 50 * (hoverType - 9) + 25)
            else
                text = titleGroup.childAt(3,25 + (hoverType - 1) * 50)
            if(text)
                text.color = "#8a8a8a"
           //  toolbox.state = "collapsed";


        }
    }

    function onLive()
    {        
		if(root.panoMode === 2)
		{
			root.onChangePanoMode();
		}
		else
		{
			root.curMode = 1
			qmlMainWindow.setCurrentMode(root.curMode);
			clearTopCtrl();
			clearSelected()
//			initLiveTopControls();
		}
    }

    function onSpherical() {
        root.curMode = 2
        qmlMainWindow.setCurrentMode(root.curMode);
        clearTopCtrl();
        clearSelected()
//        initSphericalTopControls();
    }

    function onInterative() {
        root.curMode = 3
        qmlMainWindow.setCurrentMode(root.curMode);
        clearTopCtrl();
        clearSelected();
//        initInteractiveTopControls();
    }
    function setPauseMode() {
        qmlMainWindow.setPlayMode(1);
        startImage.visible = true;
        pauseImage.visible = false;
        pauseText.text = qsTr("Start");
    }


    function setPlayMode(){
        qmlMainWindow.setPlayMode(2);
        pauseImage.visible = true;
        startImage.visible = false;
        pauseText.text = qsTr("Pause");
    }

	function closeProject()
	{
		root.panoMode = 2;
		root.onChangePanoMode();
		var isDisconnected = qmlMainWindow.disconnectComplete();
		if (isDisconnected)
			return;
        statusItem.setPauseMode();
		maskID.visible = true;
        busyLabel.text = "Closing...";
        root.isWait = false;
        qmlMainWindow.setCloseProject(true);
        qmlMainWindow.disconnectProject();
        liveView.destroyLiveCameraViews();
        root.setCurrentTitle("");
        statusItem.setStreamingItem(false)
	}

    function onNotifyHistory()
    {
        if(notifyHistorybox.isOpen){
             notifyHistorybox.state = "collapsed";
            notifyHistorybox.isOpen = false;
        }else{
            notifyHistorybox.state = "expanded";
            notifyHistorybox.isOpen = true;
        }

        notifyHistorybox.getNotifyHistory();
    }

    function onSave() {
        statusItem.setPauseMode();

        if(root.isTemplate) {
            toolbox.saveIniFile()
        }else{
            qmlMainWindow.saveIniPath(qsTr(""))
        }
        statusItem.setPlayAndPause();
    }


    function onSaveAs() {
        statusItem.setPauseMode();
        clearSelected()
        saveINIFileDialog.open();
    }

	function onReloadConfig() {
		reloadFileDialog.open();
	}

    FileDialog {
        id: saveINIFileDialog
        selectExisting: false
        selectFolder: false
        selectMultiple: false
        nameFilters: [ "L3D file(*.l3d)"]
        selectedNameFilter: "All files (*)"
        onAccepted: {
            root.isTemplate = false;
            var fileName = fileUrl.toString().substring(8);
            var index = qmlMainWindow.saveIniPath(fileName);
            root.addRecentList(index);
            toolBox.clearSelected();
            statusItem.setPlayAndPause();
            root.setCurrentTitle(qmlMainWindow.getRecentTitle(0) + ".l3d");

            qmlMainWindow.setStreamingPath(streamingbox.getStreamingPath())
        }
        onRejected: {
            toolBox.clearSelected();
            statusItem.setPlayAndPause();
        }

    }

	FileDialog {
        id: reloadFileDialog
        title: "Select configuration file"
        selectFolder: false
        nameFilters: [ "L3D file(*.l3d)"]
        selectedNameFilter: "All files (*)"
        onAccepted: {
            root.isTemplate = false;
			root.openIniProject (fileUrl);            
			toolBox.clearSelected();			
        }
        onRejected: {
            toolBox.clearSelected();            
        }
    }

    function saveIniFile(){
        saveINIFileDialog.open()
    }

    function onSettings() {
        qmlMainWindow.resetConfigList();
        //root.isTemplate = false;
        isSetting = true;

        statusItem.setPauseMode();
        clearSelected()

        var captureType = qmlMainWindow.getCaptureType();
        switch(captureType)
        {
        case TemplateModes.LIVE:
            dshowSettingbox.state = "expanded";
            dshowSettingbox.setChangeTitle();
            dshowSettingbox.updateLiveSlot();
            break;
        case TemplateModes.VIDEO:
            videoSettingbox.isFirstOpened = false
            videoSettingbox.state = "expanded";
            videoSettingbox.setChangeTitle();
            videoSettingbox.updateVideoSlot();
            break;
        case TemplateModes.IMAGE:
            imageSettingbox.state = "expanded";
            imageSettingbox.setChangeTitle();
            imageSettingbox.updateImageSlot();
            imageSettingbox.getResolution();
            break;
        }

        root.clearBackForward();

        sphericalWindow.clearSphericalWindow();
    }

    function clearHovered() {
        hoveredRectangle.y = -200
    }

    function clearSelected() {
        selectedRectangle.y = liveMouseArea.y
        if(toolbox.state === "expanded")
            toolbox.state = "collapsed"

        if(toolbox.state === "collapsed")
            isSelected = true
        else
            isSelected = false

        titleText.text = ""

        liveBox.state = "collapsed"
		sphericalBox.collapseSphericalbox();
        interactiveBox.state = "collapsed";
        snapshotBox.state = "collapsed";
        notifyHistorybox.state = "collapsed";
        sphericalView.initBanners();
        oculusGroup.state = "collapsed";
        groupCtrl.state = "collapsed";
        exposureGroup.state = "collapsed";
        weightmapTopCtrl.state = "collapsed";
        root.onCloseWeightmap();
		root.onCloseCTSettingbox();
		root.onCloseStitchCameraTemplateSettingbox();
		root.onCloseLutCtrl();

        switch(root.curMode){
        case 1:     // Live mode
            selectedRectangle.y = liveMouseArea.y
            titleText.text = liveText.text;

            liveView.visible = true;
            sphericalView.visible = false;
            interactiveView.visible = false;

            initLiveTopControls();

            break;
        case 2:     // Spherical mode
            selectedRectangle.y = sphericalMouseArea.y - 1
            titleText.text = sphericalText.text;
            sphericalView.visible = true;
            liveView.visible = false;
            interactiveView.visible = false;

            initSphericalTopControls();

            break;
        case 3:     // Interactive mode
            selectedRectangle.y = interativeMouseArea.y - 1
            titleText.text = interactiveText.text;
            interactiveView.visible = true;
            liveView.visible = false;
            sphericalView.visible = false;

            initInteractiveTopControls();
        }
    }

    function clearDisabled() {
        disableRectangle.y = -200
    }

    function showSpecifyBox() {
        if(root.curMode === 1 || root.curMode === 2){
            snapshotBox.state = "expanded";
        }
    }
    function showMainBox() {
        switch(root.curMode) {
        case 1:
            if(liveBox.state == "collapsed")
                liveBox.state = "expanded"
            liveBox.getLensSettings();
            break;
        case 2:

            if(sphericalBox.state === "collapsed")
                statusItem.setPauseMode();
            qmlMainWindow.setTempCameraSettings();
            sphericalBox.getExposureSetting();
            sphericalBox.getBlendAndParams();
            sphericalBox.isExposure = false;
            sphericalBox.showBlendSettings();
            sphericalBox.createTopCameraCombo();
            sphericalBox.getCameraList();
            sphericalBox.state = "expanded"
            break;

        case 3:
            if(interactiveBox.state === "collapsed")
                statusItem.setPauseMode();;
            interactiveBox.getOculus();
            interactiveBox.state = "expanded"
            break;
        default:
            break;
        }
    }

    function clearSettingBox()
    {
        snapshotCtrl.visible = true;
		cameraTemplateCtrl.visible = true;
		calibrationCtrl.visible = true;
        moreCtrl.visible = true;
        hoverType = 2;
        root.curMode = 1;

    }

   function clearTopCtrl(){
       moreCtrl.visible = false;
       snapshotCtrl.visible = false;
	   cameraTemplateCtrl.visible = false;
	   calibrationCtrl.visible = false;
       exposureCtrl.visible = false;
       ptsAndpacCtrl.visible = false;
       groupCtrl.visible = false;
       oculusSwitch.visible = false;
       bannerItem.visible = false;
       screenNumListview.visible = false;
       weightmapItem.visible = false;
       panoSnapshotItem.visible = false;
	   lutCtrl.visible = false;
	   ctCtrl.visible = false;
	   stitchCameraTemplateCtrl.visible = false;
   }

   function initLiveTopControls(){
       moreCtrl.visible = true;
       snapshotCtrl.visible = true;
	   cameraTemplateCtrl.visible = true;
	   calibrationCtrl.visible = true;
       exposureCtrl.visible = false;
       ptsAndpacCtrl.visible = false;
       groupCtrl.visible = false;
       oculusSwitch.visible = false;
       bannerItem.visible = false;
       screenNumListview.visible = false;
       weightmapItem.visible = false;
       panoSnapshotItem.visible = false;
	   lutCtrl.visible = false;
	   ctCtrl.visible = false;
	   stitchCameraTemplateCtrl.visible = false;
   }

   function initSphericalTopControls(){
       oculusSwitch.visible = false;
       snapshotCtrl.visible = false;
	   cameraTemplateCtrl.visible = false;
	   calibrationCtrl.visible = false;
       exposureCtrl.visible = true;
       ptsAndpacCtrl.visible = true;
       moreCtrl.visible = true
       bannerItem.visible = true;
       screenNumListview.visible = true;
       weightmapItem.visible = true;
       panoSnapshotItem.visible = true;
	   lutCtrl.visible = true;
	   ctCtrl.visible = true;
	   stitchCameraTemplateCtrl.visible = true;

	   if(root.panoMode === 2)
	   {
			exposureCtrl.visible = false;
			ptsAndpacCtrl.visible = false;
			moreCtrl.visible = false
			bannerItem.visible = false;
			weightmapItem.visible = false;
			panoSnapshotItem.visible = false;
			lutCtrl.visible = false;
			ctCtrl.visible = false;
			stitchCameraTemplateCtrl.visible = false;
			screenNumListview.anchors.right = moreCtrl.anchors.right;
		} else {
			screenNumListview.anchors.right = weightmapItem.left;
		}
   }

   function initInteractiveTopControls(){
       oculusSwitch.visible = true;       
       moreCtrl.visible = false;
       snapshotCtrl.visible = false;
	   cameraTemplateCtrl.visible = false;
	   calibrationCtrl.visible = false;
       exposureCtrl.visible = false;
       ptsAndpacCtrl.visible = false;
       groupCtrl.visible = false;
       bannerItem.visible = false;
       screenNumListview.visible = false;
       weightmapItem.visible = false;
       panoSnapshotItem.visible = false;
	   lutCtrl.visible = false;
	   ctCtrl.visible = false;
	   stitchCameraTemplateCtrl.visible = false;
   }

   function changeFromPanoMode(){
		loadGroupToolboxImage();
		changeLiveText();
		initSphericalTopControls();
   }

   function loadGroupToolboxImage(){
		if(root.panoMode === 1)
		{
			groupToolBoxImage.source = "../resources/btn_group_toolbox.png";
			bottomGroupToolBoxImage.visible = true;
			settingText.visible = true;
		}
		else
		{
			groupToolBoxImage.source =  "../resources/btn_group_toolbox_playback.png";
			bottomGroupToolBoxImage.visible = false;
			settingText.visible = false;
		}
   }

   function changeLiveText(){
		if(root.panoMode === 1)
			liveText.text =  "Live"
		else
			liveText.text =  "Quit Playback"
   }
}
