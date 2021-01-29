import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2
import Look3DQmlModule 1.0
import "../controls"
import "../components"

Item {
    id: liveSettingbox
    width : 1030
    height: 684
    opacity: 1


    property int        camCount
    property int        audioCount
    property int        camIndex: 0
    property int        audioIndex: 0
    property bool       isCameraSelected: false
    property bool       isAudioSelected: false
    property int        panorama_Mono: 0
    property int        panorama_LeftEye: 1
    property int        panorama_RightEye: 2
    property int        panorama_BothEye: 3
    property int        mixedChannel: 0
    property int        leftChannel: 1
    property int        rightChannel: 2
    property int        noAudio: 3
    property var        audioName
    property var		slotInfoList: []

    property int		deviceType: 0		// Default: "DirectShow"
    property string 	nodalWeightDescription: "Background weight map"
    property int        foregroundSlotIndex_: 0

    MouseArea {
        anchors.fill: parent
    }

    TemplateLoadSettingbox{
       id: templateLoadSettingbox
       z:  10
       x: root.width * 0.8
       visible:  false
    }

    Rectangle {
        id: titleRectangle
        width: parent.width
        height: 48
        color: "#1f1f1f"
        opacity: 1
        z: 1
        Text {
            id: titleText
            x: 50
            y: (parent.height - height) / 2
            z: 0
            color: "#ffffff"
            text: (deviceType == 0) ? "Live Camera Template" : "Blackagic DeckLink Template (Beta)"
            font.bold: false
            font.pixelSize: 20
        }


        Item {
            id: groupControl
            x: root.width - groupControl.width
            width: 230
            height: 48
            z: 1

            ToolbarItem {
                id: checkItem
                anchors.right: uncheckItem.left
                z: 2
                imgUrl: "../../resources/check.png"
                title: "Create Configuration"

                onClicked: {
                    var camLock = true;
                    var selectedCameraCnt = 0;
                    var foregroundCameraCnt = 0

                    
                    var onlineNodal = false;

                    for(var i=0;i<camList.count;i++){
                        var  item = camList.get(i);

                        if(i == cameraListView.nodalIndex)
                        {
                            onlineNodal = true;
                            continue;
                        }

                        if(!item.isSelect)
                            continue;
                        else
                        {
                            camLock = false;
                            selectedCameraCnt += 1;
                            //break;

                            if (!item.isFootage) {
                                foregroundCameraCnt++
                            }
                        }
                    }

                    if (foregroundCameraCnt == 0 && camList.count >= 0)
                        camLock = true
                        //root.showNotifyMsg("Warning","Please select only one camera in Nodal Shooting mode.");

                    if(camLock === false){
                        liveSettingbox.state = "collapsed";
                        recent.state = "collapsed";
                        root.openCameraViews();
                        busyTimer.restart();
                    }
                }
            }

            Timer {
                id: busyTimer
                interval: 100
                running: false
                repeat: false
                onTriggered: openLiveConfiguration()
            }

            ToolbarItem {
                id: uncheckItem
                anchors.right: saveItem.left
                z: 2
                title: "Back"
                imgUrl: "../../resources/uncheck.png"

                onClicked: {
                    liveTempCamSetting.state = "collapsed";
                    liveSettingbox.state = "collapsed";
                    //recentListView.enabled = true;
                    qmlMainWindow.resetTempGlobalSettings();
                    camList.clear();
                    audioList.clear();
                    if(!toolbox.isSetting) return;
                    else {
                        statusItem.setPlayAndPause();
                        root.onForward();
                    }
                }
            }

            ToolbarItem {
                id: saveItem
                anchors.right: arrangeItem.left
                title: "Save Configuration"
                imgUrl: "../../resources/save_as.png"

                onClicked: {
                    setGlobalSettings();
                    tempSaveDialog.open();
                }
            }

            ToolbarItem {
                id: arrangeItem
                anchors.right: moreItem.left
                title: "Map Rig Template"
                imgUrl: "../../resources/applicationSetting.png"
                z: 2

                onClicked: {
                    templateLoadSettingbox.x = (centralItem.width - templateLoadSettingbox.width) / 2 ;
                    templateLoadSettingbox.y = 150;
                    templateLoadSettingbox.visible = true;

                    updateTemplateLoadSettingbox();
                }
            }

            ToolbarItem {
                id: moreItem
                anchors.right: parent.right
                z: 2
                title: ""
                imgUrl: "../../resources/more_control.png"

                onClicked: {
                    if(camList.count === 0) return;
                    if(slotInfoList.length === 0) return;
                    globalSettings()
                }
            }

        }
    }

    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#000000"
        z: 0
        opacity: 1.0
    }

    ListView {
        id: cameraListView
        y: 80
        width: parent.width
        height: 50 * count
        z: 1
        spacing: 0

        model: ListModel {
            id: camList
        }

//        ExclusiveGroup {
//            id: camListExclusiveGroup
//        }

        delegate: Item {
            x: 5
            width: parent.width
            height: 50
            Row {
                LiveCamListitem {
                    title: titleText
                    titleTextColor: titleColor
                    checkSelect: isSelect
                    footageImagePath: footageImageName
                    stereoLColor: stereoLeftColor
                    stereoRColor: stereoRightColor
                    liveStereoType: stereoType
                    isStereoLeft: selectStereoLeft
                    isStereoRight: selectStereoRight
                    checkFootage: isFootage
                }
            }
        }

        function updateFootageImagePath(index,name){
            camList.set(index ,{"footageImageName": name});
        }

        function setCheckFootage(index, checkFootage) {
            camList.set(index, {"isFootage": checkFootage})
        }
    }

    Spliter{
        id: split
        width: root.width - 50
        height: 1
        anchors.left: parent.left
        anchors.leftMargin: 30
        anchors.top: cameraListView.bottom
        anchors.topMargin: 5
        anchors.right: parent.right
        anchors.rightMargin: 30
    }

    ListView {
        id: audioListView
        y: cameraListView.height + 90
        anchors.top: split.bottom
        anchors.topMargin: 20
        width: parent.width
        height: 50 * count
        z: 1
        spacing: 0
        model: ListModel {
            id: audioList
        }
        delegate: Item {
            x: 5
            width: parent.width
            height: 50
            Row {
                LiveAudioitem {
                    title: audioText
                    titleTextColor: titleColor
                    checkSelect: isSelect
                    liveAudioType: audioType
                    isAudioLeft: selectAudioLeft
                    isAudioRight: selectAudioRight
                    audioLeftColor: leftColor
                    audioRightColor: rightColor
                }
            }
        }
   }

    Rectangle {
        id: nodalRectangle
        width: parent.width - 40
        height: 20
        anchors.top: audioListView.bottom
        anchors.right: parent.right
        anchors.rightMargin: 25
        anchors.left: parent.left
        anchors.leftMargin: 25
        color: "#1f1f1f"

        Text {
            x: 35
            y: (parent.height - height) / 2
            color: "#ffffff"
            text: "Nodal Shooting (Beta)"
            font.bold: false
            font.pixelSize: 12
        }
    }

    ListView {
        id: nodalListView
        y: audioListView.height + 90
        anchors.top: nodalRectangle.bottom
        anchors.topMargin: 20
        width: parent.width
        height: 50 * count
        z: 1
        spacing: 0
        model: ListModel {
            id: nodalList
        }
        property bool enableFootage: true
        delegate: Item {
            x: 5
            width: parent.width
            height: 50
            Row {
                NodalListitem {
                    videofilePath: videofileName
                    imagefilePath: imagefileName
                    titleTextColor: titleColor
                    checkSelect: isSelect

                }
            }
        }

        function initfootageVideo(){
            nodalList.set(0 ,{"videofileName": "Background footage video"});
        }

        function updatefootageVideo(videoPath){
            nodalList.set(0 ,{"videofileName": videoPath});
        }

        function initfootageImage(){
            nodalList.set(0 ,{"imagefileName": liveSettingbox.nodalWeightDescription});
        }

        function updatefootageImage(imagePath){
            nodalList.set(0 ,{"imagefileName": imagePath});
        }
   }

    FileDialog {
        id: nodalShootvideoDialog
        title: "Please choose a background footage video file"
        nameFilters: [ "Video file (*.mp4)", "All files (*)" ]
        onAccepted: {
            var videoPath = fileUrl.toString().substring(8); // Remove "file:///" prefix
            nodalList.set(0 ,{"videofileName": videoPath})
        }
        onRejected: {
            nodalList.set(0 ,{"videofileName": "Background footage video"})
        }
    }

    FileDialog {
        id: nodalShootimageDialog
        title: "Select mask image of background footage"
        nameFilters: [ "Image file (*.jpg *.bmp *.png)", "All files (*)" ]
        selectMultiple: true
        onAccepted: {
            var imagePath = fileUrl.toString().substring(8); // Remove "file:///" prefix
            nodalList.set(0 ,{"imagefileName": imagePath});
        }
        onRejected: {
            nodalList.set(0 ,{"imagefileName": liveSettingbox.nodalWeightDescription});
        }
    }

    FileDialog {
        id: footageimageDialog
        title: "Select mask image of background footage"
        nameFilters: [ "Image file (*.jpg *.bmp *.png)", "All files (*)" ]
        selectMultiple: true
        property int camIndex: 0
        onAccepted: {
            var imagefullPath = fileUrl.toString().substring(8); // Remove "file:///" prefix
            cameraListView.updateFootageImagePath(camIndex,imagefullPath);
        }
        onRejected: {
            cameraListView.updateFootageImagePath(camIndex, liveSettingbox.nodalWeightDescription);
        }
    }

    LiveTempCamSetting {
        id: liveTempCamSetting
        anchors.right: parent.right
        anchors.rightMargin: 100
        width: 200
        height: 0
        z: 1
        state: "collapsed"

           states: [
               State {
                   name: "collapsed"
                   PropertyChanges { target: liveTempCamSetting; height: 0}
                   PropertyChanges { target:  liveTempCamSetting;width: 0

                   }
               },
               State {
                   name: "expanded"
                   PropertyChanges { target: liveTempCamSetting; height: 300}
                   PropertyChanges {target: liveTempCamSetting;width: 200}
               }
           ]

           transitions: [
               Transition {
                   NumberAnimation { target: liveTempCamSetting; property: "height"; duration: 300 }
                   NumberAnimation { target: liveTempCamSetting;property: "width";duration: 300}
               }
           ]

    }

    VideoGlobalSettings {
        id: globalSettingbox
        height: root.height - 32
        width: 350
        z: 1
        state: "collapsed"

           states: [
               State {
                   name: "collapsed"
                   PropertyChanges { target:  globalSettingbox; x: root.width}
               },
               State {
                   name: "expanded"
                   PropertyChanges { target: globalSettingbox; x: root.width - 350}
               }
           ]

           transitions: [
               Transition {
                   NumberAnimation { target: globalSettingbox;property: "x";duration: 200 }
               }
           ]

    }

    FileDialog {
        id: tempSaveDialog
        selectExisting: false
        selectFolder: false
        selectMultiple: false
        nameFilters: [ "L3D file(*.l3d)"]
        selectedNameFilter: "All files (*)"
        onAccepted: {
            var fileName = fileUrl.toString().substring(8);
            var index = qmlMainWindow.saveIniPath(fileName);
            root.addRecentList(index);
        }
    }

   function setCurrentCamera() {
     camListModel.set(videoIndex,{"titleText": videoPath})
   }

   property var isNextProjectQueued: false

   function openLiveConfiguration()
   {
        //root.openCameraViews();
        if (qmlMainWindow.isEmpty)
        {
            onOpenProject();
        }
        else
        {
            isNextProjectQueued = true;
            toolbox.closeProject();
        }
   }

   function previousConfigurationClosed()
    {
        if (isNextProjectQueued)
        {
            isNextProjectQueued = false;
            onOpenProject();
        }
    }

   function onOpenProject(){
       root.initVariants();
       root.initUI();
       setGlobalSettings();

       qmlMainWindow.openProject();
       root.createCameraViews();

       toolbox.clearSelected();

       root.panoMode = 2;
       root.onChangePanoMode();
   }

   // If video on nodalListVeiw is selected as background, Live cameras can not be selected as a backgroundRectangle
   function setGlobalSettings() {
       qmlMainWindow.initTemplateCameraObject()

       var isLiveNodal = false

       if (nodalListView.enableFootage)
       {
           var nodalVideo = nodalList.get(0)
           if (nodalVideo.videofileName != "Background footage video") {
               qmlMainWindow.setNodalVideoFilePath(0, nodalList.get(0).videofileName)
               qmlMainWindow.setNodalMaskImageFilePath(0, nodalList.get(0).imagefileName)
               qmlMainWindow.setNodalSlotIndex(0)
               qmlMainWindow.setNodalCameraIndex(-1)
           }
       } else {
            for (var i = 0; i < camList.count; i ++) {
                var cameraItem_ = camList.get(i)

                if (!cameraItem_.isFootage) continue;
                if (!cameraItem_.isSelect) continue;

                qmlMainWindow.setNodalVideoFilePath(i, cameraItem_.devicePath)
                qmlMainWindow.setNodalMaskImageFilePath(i, cameraItem_.footageImageName)
                qmlMainWindow.setNodalCameraIndex(i)
                qmlMainWindow.setNodalSlotIndex(i)

                // 10: selected and footage
                qmlMainWindow.setTotalIndexAndSelectedIndex(i, 10)
            }

            isLiveNodal = true
       }

       var cameraCnt = 0;
       for(var camIndex = 0; camIndex < camList.count; camIndex++){
           var cameraItem = camList.get(camIndex);

           if (!cameraItem.isSelect) continue
           if (!isLiveNodal)
               if (cameraItem.isFootage) continue

           qmlMainWindow.sendCameraName(camIndex, cameraItem.devicePath, false, false)
           qmlMainWindow.setTempStereoType(camIndex, cameraItem.stereoType)
           qmlMainWindow.setForegroundSlot(camIndex)
           cameraCnt ++

           // 11: selected and not footage
           if (cameraItem.isFootage) continue
           qmlMainWindow.setTotalIndexAndSelectedIndex(camIndex, 11)
       }

       // total live camera setting
       for (var totalIndex = 0; totalIndex < camList.count;  totalIndex++ ) {
           var cameraItem__ = camList.get(totalIndex)
           qmlMainWindow.setLiveCamera(totalIndex, cameraItem__.devicePath)
           qmlMainWindow.setLiveStereo(totalIndex, cameraItem__.stereoType)

           if (cameraItem__.isSelect && cameraItem__.isFootage) {
               // 12: not select and not footage
               qmlMainWindow.setTotalIndexAndSelectedIndex(camIndex, 12)
           }
       }

       qmlMainWindow.setCameraCnt(cameraCnt)

       for(var audioIndex=0, selCount=0; audioIndex < audioList.count; audioIndex++) {
           var audioItem = audioList.get(audioIndex);
           if(!(audioItem.isSelect && selCount+1 <= camList.count)) continue;
           qmlMainWindow.sendAudioName(audioIndex,audioItem.audioText);
           qmlMainWindow.setTempAudioSettings(audioIndex,audioItem.audioType);
           selCount++;
       }
       
       qmlMainWindow.openTemplateCameraIniFile(getSlotListByString(), templateLoadSettingbox.getOrderList());
   }

   function globalSettings(){
		
       globalSettingbox.changeLiveMode();
       liveTempCamSetting.state = "collapsed";
       globalSettingbox.getGlobalValues();
       if(globalSettingbox.state == "expanded"){
           globalSettingbox.state = "collapsed";
       }else if(globalSettingbox.state == "collapsed"){
           globalSettingbox.state = "expanded";
       }
   }

   function setCameraSettings(index){
       liveTempCamSetting.setCameraValues(index);
   }

   function getCameraSettings(index)
   {
       liveTempCamSetting.getCameraValues(index);
   }

   function getCameraValues(camIndex)
   {
       var type =  qmlMainWindow.getTempStereoType(camIndex)
       switch(type){
       case 0:
           camList.set(camIndex,{"stereoLeftColor": "#8a8a8a"})
           camList.set(camIndex,{"stereoRightColor": "#8a8a8a"})
           break;
       case 1:
           camList.set(camIndex,{"stereoLeftColor": "#8a8a8a"})
           camList.set(camIndex,{"stereoRightColor": "#8a8a8a"})
           break;
       case 2:
           camList.set(camIndex,{"stereoLeftColor": "#8a8a8a"})
           camList.set(camIndex,{"stereoRightColor": "#8a8a8a"})
           break;
       case 3:
           camList.set(camIndex,{"stereoLeftColor": "#8a8a8a"})
           camList.set(camIndex,{"stereoRightColor": "#8a8a8a"})
           break;
       default:
           break;
       }
   }

   function initLiveSlots()
   {
       camList.clear();
       camCount = qmlMainWindow.getCameraCnt();
       for(var i = 0;i<camCount;i++){
           camList.set(i ,{"titleText": qmlMainWindow.getCameraDeviceName(i)});
           camList.set(i ,{"devicePath": qmlMainWindow.getCameraDevicePath(i)});
          // camList.set(i ,{"titleText": "webCam" + i});

           camList.set(i,{"titleColor": "#ffffff"})
           camList.set(i ,{"isSelect": true});
           camList.set(i ,{"isFootage": false});
           camList.set(i ,{"footageImageName": liveSettingbox.nodalWeightDescription});
           camList.set(i,{"stereoLeftColor": "#8a8a8a"})
           camList.set(i,{"stereoRightColor": "#8a8a8a"})
           camList.set(i,{"selectStereoLeft": false})
           camList.set(i,{"selectStereoRight": false})
           camList.set(i,{"stereoType": 0});

           setStereoSettings(i);
       }

       nodalListView.enableFootage = true;

       audioCount = qmlMainWindow.getAudioCnt();
       //audioCount = 2;
       if( audioCount === 0) {
           split.visible = false;
       }
       else if( audioCount > 0){
           split.visible = true;
       }

       audioList.clear();
       for(var j = 0; j<audioCount;j++){
           audioList.set(j ,{"audioText": qmlMainWindow.getMicDeviceName(j)});
           //audioList.set(j ,{"audioText": "MicOne"});
           audioList.set(j,{"titleColor": "#8a8a8a"})
           audioList.set(j ,{"audioType": 1});
           audioList.set(j ,{"selectAudioLeft": true});
           audioList.set(j ,{"selectAudioRight": false});
           audioList.set(j ,{"leftColor": "#ffffff"});
           audioList.set(j ,{"rightColor": "#8a8a8a"});
           audioList.set(j ,{"isSelect": false})
       }
       if(camList.count === 0){
           nodalRectangle.visible = false;
           return;
       }
       getInfoStr();

       //nodal shoot
       nodalList.clear();
       nodalRectangle.visible = true;
       nodalListView.initfootageVideo();
       nodalListView.initfootageImage();
       nodalList.set(0,{"titleColor": "#ffffff"})
       nodalList.set(0 ,{"isSelect": false});

       checkTemplateMode();

       updateTemplateLoadSettingbox();
   }

   function clearFootage(i){
       camList.set(i, {"isFootage": false});
   }

   function getInfoStr()
   {
       var camName = camList.get(foregroundSlotIndex_).devicePath
       slotInfoList = qmlMainWindow.getSlotInfo(camName,"",TemplateModes.LIVE);
   }

   function setChangeTitle()
   {
       titleText.text = "Live Camera Setting"
   }

   function updateLiveSlot()
   {
       var cameraName
       var cameraPath
       var nodalSlotIndex

       camList.clear()
       camCount = qmlMainWindow.getLiveCameraCount()

       var statusIndex;
       for (var j = 0; j < camCount; j++) {
           statusIndex = qmlMainWindow.getSelectedIndex(j)

           if (statusIndex == 12) { // not select and not footage
               camList.set(j, {"isFootage": false})
               camList.set(j, {"titleText": qmlMainWindow.getCameraDeviceName(j)})
               camList.set(j ,{"isSelect": false});
           } else if (statusIndex == 11) // select and not footage
           {
               camList.set(j, {"isFootage": false})
               camList.set(j, {"titleText": qmlMainWindow.getCameraDeviceName(j)})
               camList.set(j ,{"devicePath": qmlMainWindow.getCameraDevicePath(j)});
               camList.set(j ,{"isSelect": true});

           } else if(statusIndex == 10) { // select and footage
               camList.set(j, {"isFootage": true})
               camList.set(j, {"titleText": qmlMainWindow.getCameraDeviceName(j)})
               camList.set(j ,{"devicePath": qmlMainWindow.getCameraDevicePath(j)});
               camList.set(j ,{"isSelect": true});
           }

           camList.set(j, {"titleColor": "#ffffff"})
           camList.set(j ,{"footageImageName": qmlMainWindow.getNodalMaskImageFilePath(j)})
           //camList.set(j ,{"isSelect": true});
           camList.set(j, {"stereoLeftColor": "#8a8a8a"})
           camList.set(j, {"stereoRColor": "#8a8a8a"})
           camList.set(j, {"selectStereoLeft": false})
           camList.set(j, {"selectStereoRight": false})
           camList.set(j,{"stereoType": 0})
           camList.set(j,{"stereoRightColor": "#8a8a8a"})
       }



       checkTemplateMode();
       getInfoStr();

       // Audio
       audioList.clear();
       for (var j = 0; j<qmlMainWindow.getAudioCnt(); j++)
       {
           audioName = qmlMainWindow.getMicDeviceName(j);
           audioList.set(j ,{"audioText": audioName});
           isAudioSelected = qmlMainWindow.isSelectedAudio(audioName);
           if(isAudioSelected)
               audioList.set(j,{"titleColor": "#ffffff"});
           else
              audioList.set(j,{"titleColor": "#8a8a8a"});
           audioList.set(j ,{"isSelect": isAudioSelected});
		   audioList.set(j ,{"audioType": 1});
           audioList.set(j ,{"selectAudioLeft": true});
           audioList.set(j ,{"selectAudioRight": false});
           audioList.set(j ,{"leftColor": "#ffffff"});
           audioList.set(j ,{"rightColor": "#8a8a8a"});
           setAudioSettings(j);
       }

       //nodal shoot
       nodalList.clear();
       nodalRectangle.visible = true;
       nodalListView.updatefootageVideo("Background footage video");
       nodalListView.updatefootageImage(qmlMainWindow.getNodalMaskImageFilePath(0));
       nodalList.set(0,{"titleColor": "#ffffff"})
       nodalList.set(0 ,{"isSelect": false});

       nodalListView.enableFootage = true;
       if (qmlMainWindow.getNodalVideoIndex() >= 0)
            nodalListView.enableFootage = false;
       if (qmlMainWindow.getNodalVideoFilePath(0) === "Background footage video")
            nodalListView.enableFootage = false;
   }

   function updateTemplateSettingUI()
   {
       initLiveSlots();
   }

   function checkTemplateMode()
   {
       if (root.isRigTemplateMode())
       {
            arrangeItem.visible = true;
            saveItem.anchors.right = arrangeItem.left;
       }
       else
       {
           arrangeItem.visible = false;
           saveItem.anchors.right = moreItem.left;
       }
   }

   function setStereoSettings(camIndex)
   {
       var stereoType =  qmlMainWindow.getTempStereoType(camIndex)
       switch(stereoType){
       case panorama_Mono:
           camList.set(camIndex,{"stereoLeftColor": "#8a8a8a"})
           camList.set(camIndex,{"stereoRightColor": "#8a8a8a"})
           camList.set(camIndex,{"selectStereoLeft": false})
           camList.set(camIndex,{"selectStereoRight": false})
           camList.set(camIndex,{"stereoType": 0});
           break;
       case panorama_LeftEye:
           camList.set(camIndex,{"stereoLeftColor": "#ffffff"})
           camList.set(camIndex,{"stereoRightColor": "#8a8a8a"})
           camList.set(camIndex,{"stereoType": 1});
           camList.set(camIndex,{"selectStereoLeft": true})
           camList.set(camIndex,{"selectStereoRight": false})
           break;
       case panorama_RightEye:
           camList.set(camIndex,{"stereoLeftColor": "#8a8a8a"})
           camList.set(camIndex,{"stereoRightColor": "#ffffff"})
           camList.set(camIndex,{"stereoType": 2});
           camList.set(camIndex,{"selectStereoLeft": false})
           camList.set(camIndex,{"selectStereoRight": true})
           break;
       case panorama_BothEye:
           camList.set(camIndex,{"stereoLeftColor": "#ffffff"})
           camList.set(camIndex,{"stereoRightColor": "#ffffff"})
           camList.set(camIndex,{"stereoType": 3});
           camList.set(camIndex,{"selectStereoLeft": true})
           camList.set(camIndex,{"selectStereoRight": true})
           break;
       default:
           break;
       }
   }

  function setAudioSettings()
  {
      var audioType =  qmlMainWindow.getTempAudioSettingsEx(audioName)
      switch(audioType){
      case mixedChannel:
          audioList.set(audioIndex,{"leftColor": "#ffffff"})
          audioList.set(audioIndex,{"rightColor": "#ffffff"})
          audioList.set(audioIndex ,{"audioType": 0});
          audioList.set(audioIndex,{"selectAudioLeft": true})
          audioList.set(audioIndex,{"selectAudioRight": true})
          break;
      case leftChannel:
          audioList.set(audioIndex,{"leftColor": "#ffffff"})
          audioList.set(audioIndex,{"rightColor": "#8a8a8a"})
          audioList.set(audioIndex ,{"audioType": 1});
          audioList.set(audioIndex,{"selectAudioLeft": true})
          audioList.set(audioIndex,{"selectAudioRight": false})
          break;
      case rightChannel:
          audioList.set(audioIndex,{"leftColor": "#8a8a8a"})
          audioList.set(audioIndex,{"rightColor": "#ffffff"})
          audioList.set(audioIndex ,{"audioType": 2});
          audioList.set(audioIndex,{"selectAudioLeft": false})
          audioList.set(audioIndex,{"selectAudioRight": true})
          break;
      case noAudio:
          audioList.set(audioIndex,{"leftColor": "#8a8a8a"})
          audioList.set(audioIndex,{"rightColor": "#8a8a8a"})
          audioList.set(audioIndex ,{"audioType": 3});
          audioList.set(audioIndex,{"selectAudioLeft": false})
          audioList.set(audioIndex,{"selectAudioRight": false})
          break;
      default:
          break;
      }
  }

  function updateTemplateLoadSettingbox()
  {
      var index = 0;
      var curSlotList = [];
      for(var i = 0 ; i < camList.count ; i++){
          var  item = camList.get(i);
          if(item.isSelect) {
            curSlotList[index] = item.devicePath;
            index++;
          }
      }
      templateLoadSettingbox.updateUI(1, curSlotList);
  }

  function getSlotListByString()
  {
      var strSlotList = [];
      for(var i = 0 ; i < camList.count; i++){
          var  item = camList.get(i);
          if(item.isSelect)
              strSlotList.push(item.devicePath);
      }
      return strSlotList;
  }

}
