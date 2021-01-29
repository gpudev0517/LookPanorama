import QtQuick 2.4
import QtQuick.Controls 1.3
import QtQuick.Window 2.2
import QtQuick.Dialogs 1.2
import QtQml 2.2
import QtQuick.Layouts 1.1
import MCQmlCameraView 1.0
import "LiveCamViewPosition.js" as LiveCamViewPosition
import "./"

Item {
    id: liveArea
    anchors.fill: parent
    visible: false
    property int cellWidthVal: 385
    property int cellHeightVal: 320
    property int topMargin: 20
    property var colors: ["Red","Green","Blue","Dark cyan","Magenta","#808000","Dark gray","Dark red","Dark green","Dark blue","Dark magenta","Gray","Light gray"]
    property var componentList: []
    property var liveCamViewList: []
    property var cpGroupList: []
    property int frameWidth: 400
    property int frameHeight: 300
    property bool isCalibVisibled: false    
    property int liveCam_index:0
    property bool isMaximized: false
    property int liveCamCount: 8
    CameraTemplateSettingbox{
           id: cameraTemplateSettingbox
           z:  10
           x: root.width
           visible:  false
    }

    Rectangle {
        id: backgroundRectangle
        color: "#000000"
        anchors.fill: parent

        ScrollView {
            id: liveScrollView
            anchors.fill:  parent
            verticalScrollBarPolicy: Qt.ScrollBarAlwaysOff
            horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff

            Rectangle {
                id: liveCamView
                property int scrollSize: 30
                width: 0
                height: 0
                color: "#000000"
            }
        }

        onHeightChanged:  {
            resetLiveCamViewSize();
            LiveCamViewPosition.setVideoWindowsGeom(liveCamView.children, liveArea, cellWidthVal, cellHeightVal);
            LiveCamViewPosition.setSelectedLiveCamViewGeom(liveCamView.children, liveArea);
        }

        onWidthChanged:   {
            resetLiveCamViewSize();
            LiveCamViewPosition.setVideoWindowsGeom(liveCamView.children, liveArea, cellWidthVal, cellHeightVal);
            LiveCamViewPosition.setSelectedLiveCamViewGeom(liveCamView.children, liveArea);
        }

        Component.onCompleted: {
            createLiveCameraViews();
        }
    }

    CalibrationSettingbox {
           id: calibrationSettingbox
           z:  -1
           x: root.width
           visible:  isCalibVisibled
    }

    ExposureDialog{
            id: detailsDialog
            y:  (root.height - height) / 2
            x:  (root.width - width) / 2
            z:  10
    }

    Timer {
        id: cPointTimer
        interval: 100
        running: false
        repeat: false
        onTriggered: createCPoint();
    }

    function createDetailsWindow()
    {
        detailsDialog.visible = true;
        detailsDialog.createTab();
		detailsDialog.updatePreview();
    }

    function initDetailsWindow(){
        detailsDialog.visible = false;
        clearCPoint();
        detailsDialog.clearAllCPoint1();
        detailsDialog.clearAllCPoint2();
        qmlMainWindow.initCPointList();
    }

	function createCameraTemplateWindow() {
		var isShown = cameraTemplateSettingbox.visible;
		cameraTemplateSettingbox.visible = !isShown;
        cameraTemplateSettingbox.initCameraParms();

		cameraTemplateSettingbox.x = centralItem.width - cameraTemplateSettingbox.width;
        cameraTemplateSettingbox.y = 0;
	}

    function destroyCameraTemplateWindow() {
        cameraTemplateSettingbox.visible = false;
    }

    function createTemplateVideoWindow()
    {
        createCameraWindow();
    }
    function createTemplateCameraWindow()
    {
        createCameraWindow();
    }
    function createTemplateImageWindow()
    {
        createCameraWindow();
    }

    function createCPoint()
    {
//        var frameWidth = 400;
//        var frameHeight = 300;
        var curWidth = frameWidth;
        var curHeight = frameHeight;
        var orgWidth = qmlMainWindow.getTempWidth();
        var orgHeight = qmlMainWindow.getTempHeight();
        if(orgHeight/orgWidth > frameHeight/frameWidth)
        {
            curHeight = frameHeight;
            curWidth = curHeight * (orgWidth / orgHeight);
        }
        else
        {
            curWidth = frameWidth;
            curHeight = curWidth * (orgHeight / orgWidth);
        }
        var cPointCount = 0;
        var componentIndex = -1;
        for(var i = 0; i<qmlMainWindow.getCameraCount(); i++)
        {
            cPointCount = qmlMainWindow.getCPointCountEx(i);
            for (var j = 0; j < cPointCount; j++)
            {
                var pos = qmlMainWindow.getCPointEx(j,i);
                var posList = pos.split(":");
                var xPos = (curWidth/orgWidth) * posList[0] + (frameWidth - curWidth) / 2;
                var yPos = (curHeight/orgHeight) * posList[1] + (frameHeight - curHeight) / 2 + 30;
                var cpGroupIndex = posList[2];
                cpGroupList[j] = cpGroupIndex
                var component = Qt.createComponent("ControlPoint.qml");
                //liveGridView.currentIndex = i;
                //var cameraItem = liveGridView.currentItem
                var liveCamView = liveCamViewList[i];
                var cPoint = component.createObject(liveCamView, {"xPos": xPos, "yPos": yPos, "cpIndex": j, "lblColor": colors[cpGroupIndex%colors.length]});
                componentIndex++
                componentList[componentIndex] = cPoint;
            }
        }
    }

    function clearCPoint()
    {
        var cPointCount = 0;
        var componentIndex = -1;
        for(var i = 0; i<qmlMainWindow.getCameraCount(); i++)
        {
            cPointCount = qmlMainWindow.getCPointCountEx(i);
            for (var j = 0; j < cPointCount; j++)
            {
                componentIndex++;
                var component;
                component = componentList[componentIndex];
                if (typeof(component.destroy) == "function")
                    component.destroy();
            }
        }
    }

    function clearLiveCamView() {
        for (var liveCamIndex = 0; liveCamIndex < liveCamViewList.length; liveCamIndex++ ) {
            var component = liveCamViewList[liveCamIndex];

            if (typeof(component.destroy) == "function")
                component.destroy();
        }
    }

    function updateCalibrationSettingsbox() {
        //
        calibrationSettingbox.x = centralItem.width - calibrationSettingbox.width;
        calibrationSettingbox.y = 0;
        calibrationSettingbox.z = 10;
        calibrationSettingbox.updateConfiguration();
    }

    function createCalibrationSettingbox(){
            calibrationSettingbox.visible = true;
            //weightmapSettingbox.isEditweightmap = true;
            updateCalibrationSettingsbox();
            //weightmapSettingbox.appendCameraCombo();
            //weightmapSettingbox.setDrawWeightmapSettings();
        }

    function appendCameraCombo(){
        calibrationSettingbox.camListModel.clear();

        for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
            calibrationSettingbox.camListModel.append({"text": "Camera" + (i + 1)})
        }
     }

    function closeCalibSettignBox() {
        calibrationSettingbox.visible = false
    }

   function setStopCaptureStatus() {
       calibrationSettingbox.setVisibleStopCaptureItem();
   }

   function createLiveCameraViews() {

       for (var i = 0; i < liveCamCount; i ++) {
            var component = Qt.createComponent("./MCCameraView.qml");
           if (component.status == Component.Ready) {
               var cameraObject = component.createObject(liveCamView);
               liveCamViewList[i] = cameraObject;
           }
       }
   }

   function updateLiveCameraViews(liveCamName,deviceIndex){
       qmlMainWindow.updateCameraView(liveCamName, deviceIndex)
   }

   function setLiveCamViewSize() {
       var liveCamViewGeom = LiveCamViewPosition.getLiveAreaGeom(liveCamCount, centralItem, cellWidthVal, cellHeightVal, 20);
       liveCamView.width = Math.max(liveCamViewGeom[0], liveArea.width);
       liveCamView.height = Math.max(liveCamViewGeom[1], liveArea.height);
   }

   function resetLiveCamViewSize() {
       var liveCamViewGeom = LiveCamViewPosition.getLiveAreaGeom(liveCamCount, centralItem, cellWidthVal, cellHeightVal, 20);
       liveCamView.width = Math.min(liveCamViewGeom[0], liveArea.width);
       liveCamView.height = Math.min(liveCamViewGeom[1], liveArea.height);
   }

   function setMaximizedLiveCamView() {
       liveCamView.width = centralItem.width
       liveCamView.height = centralItem.height
   }

   function initializeLiveCameraViews() {
       var viewCount = qmlMainWindow.getCameraCount();

       for(var i = 0; i < viewCount; i++) {
           var cameraObject = liveCamViewList[i];
           cameraObject.hidden = false;
           cameraObject.active = true;
           updateLiveCameraViews(cameraObject.camView,i);
       }

       for(var i = viewCount; i < liveCamCount; i++) {
           var cameraObject = liveCamViewList[i];
           cameraObject.hidden = true;
           cameraObject.active = false;
       }
   }

   function initializeLiveView() {
       updateCalibrationSettingsbox()
       setLiveCamViewSize();
       initializeLiveCameraViews();
       LiveCamViewPosition.setVideoWindowsGeom(liveCamView.children, liveArea, cellWidthVal, cellHeightVal);
   }

   function destroyLiveCameraViews()
   {
       for(var i = 0; i < liveCamCount; i++) {
           var cameraObject = liveCamViewList[i];
           cameraObject.hidden = true;
       }
   }

   function setSingleCalibMessage(message) {
        calibrationSettingbox.setMessage(message);
   }

   function setSingleCalibStrengthRatio(strengthRatio) {
       calibrationSettingbox.setStrengthRatio(strengthRatio);
   }

   function setSingleCalibStrengthDrawColor(color) {
       calibrationSettingbox.setStrengthDrawColor(color);
   }

   function setSingleCalibCaptureItem() {
       calibrationSettingbox.setCaptureItem()
   }

   function setCalibrateStatus(calibStatus) {
       calibrationSettingbox.setCalibrateStatus(calibStatus)
   }
}

