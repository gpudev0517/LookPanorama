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
import "../"

Item {
    width : 1030
    height: 684
    opacity: 1
    z: 3

    property int        videoIndex: 0
    property bool       videoLock: false
    property int        panorama_Mono: 0
    property int        panorama_LeftEye: 1
    property int        panorama_RightEye: 2
    property int        panorama_BothEye: 3
    property int        mixedChannel: 0
    property int        leftChannel: 1
    property int        rightChannel: 2
    property int        noAudio: 3
    property var		slotInfoList: []

	property var		isNextProjectQueued: false
    property bool       enableFootage: false
    property bool       isFirstOpened: true
    MouseArea {
        anchors.fill: parent
    }

    TemplateLoadSettingbox{
       id: templateLoadSettingbox
       z:  10
       x: root.width * 0.8
       visible:  false
    }    

    WeightmapSettingbox {
        id: weightMapSettingBox_
    }

    Rectangle {
        id: titleRectangle
        x: 0
        y: 0
        width: parent.width
        height: 48
        color: "#1f1f1f"
        z: 1
        opacity: 1

        Text {
            id: titleText
            x: 30
            y: (parent.height - height) / 2
            z: 3
            color: "#ffffff"
            text: qsTr("Video Template")
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
                z: 2
                anchors.right: uncheckItem.left
                imgUrl: "../../resources/check.png"
                title: "Create Configuration"

                onClicked: {

                    // initialize weightmapSettingbox
                    //weightMapSettingBox_.initWeightmapSettings()

                    // for import Multiple Videoes
                    isFirstOpened = true

                    var selectedCameraCnt = 0
                    var backgroundCameraCnt = 0
                    var videoLock = true

                    for(var j=0;j<videoList.count;j++){
                         var  item_ = videoList.get(j);
                        if(item_.isSelect) {
                            selectedCameraCnt ++
                            if (item_.isSelectedAsBackground) {
                                backgroundCameraCnt ++
                            }
                        }
                    }

                    // Normal Performance without Nodalshooting
                    if (backgroundCameraCnt == 0 && selectedCameraCnt > 0) {

                        enableFootage = false

                        for(var i=0;i<videoList.count;i++){
                             var  item = videoList.get(i);
                            if(!item.isSelect)  continue;
                            else
                            {
                                videoLock = false;
                                break;
                            }
                        }
                    }

                    // NodalShooting
                    if (backgroundCameraCnt > 0 && selectedCameraCnt > 0) {
                        enableFootage = true
                        videoLock = false
                    }

                    if ((selectedCameraCnt - backgroundCameraCnt) == 0 && videoList.count >= 0)
                        videoLock = true
                    if(videoLock === false) {
                        recent.state = "collapsed";
                        videoSettingbox.state = "collapsed";
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
                onTriggered: openVideoConfiguration()
            }

            ToolbarItem {
                id: uncheckItem
                anchors.right: plusItem.left
                z: 2
                imgUrl: "../../resources/uncheck.png"
                title: "Back"

                onClicked: {
                    isFirstOpened = true
                    slotInfoList = [];
                    cameraSettingsbox.state = "collapsed";
                    videoSettingbox.state = "collapsed";
                    //recentListView.enabled = true;
                    qmlMainWindow.resetTempGlobalSettings();

                    if(!toolbox.isSetting) return;
                    else {
                        statusItem.setPlayAndPause();
                        root.onForward();
                    }
                }
            }

            ToolbarItem {
                id: plusItem
                anchors.right: saveItem.left
                z: 2
                imgUrl: "../../resources/btn_plus.png"
                title: "Import Multiple Videos"

                onClicked: {
                    videoDialog.open();
                }

            }

            ToolbarItem {
                id: saveItem
                anchors.right: arrangeItem.left
                z: 2
                imgUrl: "../../resources/save_as.png"
                title: "Save Configuration"

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
                title: ""
                imgUrl: "../../resources/more_control.png"

                onClicked: {
                    //if(slotInfoList.length === 0) return;
                    globalSettings();
                }
            }
        }
    }

    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#000000"
        opacity: 1.0
    }

    ListView {
        id: videoListView
        x: 0
        y: 80
        width: parent.width
        height: 50 * count
        spacing: 0
        model: ListModel {
            id: videoList
        }
        delegate: Item {
            id: delegate
            x: 5
            width: parent.width
            height: 50
            Row {
                VideoListitem {
                    title: titleText
                    titleTextColor: titleColor
                    leftTextColor: leftColor
                    rightTextColor: rightColor
                    stereoLColor: stereoLeftColor
                    stereoRColor: stereoRightColor
                    checkSelect: isSelect
                    tempAudioType: audioType
                    tempStereoType: stereoType
                    isStereoLeft: selectStereoLeft
                    isStereoRight: selectStereoRight
                    isAudioLeft: selectAudioLeft
                    isAudioRight: selectAudioRight

                    footageImagePath: footageImageName
                    selectAsBackground: isSelectedAsBackground

                    onClickedFootageImage: {
                        footageimageDialog.camIndex = index;
                        footageimageDialog.open();
                    }
                }
            }
        }

        function updateFootageImagePath(index,name){
            videoList.set(index ,{"footageImageName": name});
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
            videoListView.updateFootageImagePath(camIndex,imagefullPath);
        }
        onRejected: {
            videoListView.updateFootageImagePath(camIndex, "Background weight map");
        }
    }


    CameraSettingbox {
        id: cameraSettingsbox
        anchors.right: parent.right
        anchors.rightMargin: 100
        width: 250
        height: 0
        z: 1
        state: "collapsed"

           states: [
               State {
                   name: "collapsed"
                   PropertyChanges { target: cameraSettingsbox; height: 0}
                   PropertyChanges { target:  cameraSettingsbox;width: 0

                   }
               },
               State {
                   name: "expanded"
                   PropertyChanges { target: cameraSettingsbox; height: 300}
                   PropertyChanges {target: cameraSettingsbox;width: 200}
               }
           ]

           transitions: [
               Transition {
                   NumberAnimation { target: cameraSettingsbox; property: "height"; duration: 300 }
                   NumberAnimation { target: cameraSettingsbox;property: "width";duration: 300}
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
                   PropertyChanges { target:  globalSettingbox;x: root.width}
               },
               State {
                   name: "expanded"
                   PropertyChanges { target:  globalSettingbox;x: root.width - 350}
               }
           ]

           transitions: [
               Transition {
                   NumberAnimation { target: globalSettingbox;property: "x";duration: 200 }
               }
           ]

    }

    FileDialog {
        id: videoDialog
        title: "Open Video file"
        nameFilters: [ "Video file (*.mp4)","Video file (*.mov)",  "All files (*)" ]
        selectMultiple: true
        onSelectionAccepted: {

            if (isFirstOpened) {
                for (var i = 0; i < fileUrls.length; ++i)
                {
                    videoIndex = i;
                    checkExistVideo(fileUrls[i].toString().substring(8));
                    isFirstOpened = false
                }

                getInfoStr();
                singleDlg.folder = fileUrls[0];
                globalSettingbox.setFileUrl(fileUrls[0]);
            } else {
                setVideoFiles(fileUrls)
            }
        }
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

    FileDialog {
        id: singleDlg
        title: "Open Video file"
        nameFilters: [ "Video file (*.mp4)","Video file (*.mov)", "All files (*)" ]
        selectMultiple: true
        onSelectionAccepted: {
            var videoPath = fileUrl.toString().substring(8); // Remove "file:///" prefix
            checkExistVideo(videoPath);
            getInfoStr();
            videoDialog.folder = fileUrl;
        }
    }

    function initVideoSlots(){
        videoList.clear();

        for (var i = 0; i < 12; i ++) {
            videoList.append({
                 "titleText": "Empty Slot",
                 "leftColor": "#ffffff",
                 "rightColor": "#8a8a8a",
                 "isSelect": false,
                 "selectStereoLeft": false,
                 "selectStereoRight": false,
                 "selectAudioLeft": true,
                 "selectAudioRight": false,
                 "audioType": 1,"stereoType": 0,
                 "titleColor": "#8a8a8a",
                 "stereoLeftColor": "#8a8a8a",
                 "stereoRightColor": "#8a8a8a",
                 "footageImageName": "Background weight map",
                 "isSelectedAsBackground": false
                 });
        }

        checkTemplateMode();
        enableFootage = false

    }

    function setVideoFile(videoPath){
        videoList.set(videoIndex,{"titleColor": "#ffffff"});
        videoList.set(videoIndex,{"isSelect": true});
        videoList.set(videoIndex,{"titleText": videoPath});

		updateTemplateLoadSettingbox();
        setStereoSettings();
    }

    function getInfoStr()
    {
        var videoName = videoList.get(videoIndex).titleText;
        slotInfoList = qmlMainWindow.getSlotInfo(videoName,"",TemplateModes.VIDEO);
    }

	function openVideoConfiguration()
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

        if(!root.isTemplate){
            root.setCurrentTitle(qmlMainWindow.getRecentTitle(0) + ".ini");
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
        
        //root.deleteCameraViews();
        //qmlMainWindow.disconnectProject();
		qmlMainWindow.openProject();
        root.createCameraViews();   
		     
        toolbox.clearSelected();

		root.panoMode = 2;
        root.onChangePanoMode();
    }

    function setGlobalSettings(){
        qmlMainWindow.initTemplateVideoObject();

        // Nodal Shooting is possible
        if (enableFootage) {
            for (var i=0; i<videoList.count; i++) {
                var item_  = videoList.get(i)

                if (item_.isSelectedAsBackground) {
                    var imageFileName = item_.footageImageName
                    var videoFileName = item_.titleText

                    qmlMainWindow.setNodalVideoFilePath(i, videoFileName)
                    qmlMainWindow.setNodalMaskImageFilePath(i, imageFileName)
                    qmlMainWindow.setNodalCameraIndex(-1)
                    qmlMainWindow.setNodalSlotIndex(i)
                }
            }
        }

        for(var i=0;i<videoList.count;i++){
            var item = videoList.get(i);
            if(!item.isSelect || item.titleText === "Empty slot")  continue;
            if (!item.isSelectedAsBackground) {

                qmlMainWindow.sendVideoPath(i, item.titleText);
                qmlMainWindow.setTempStereoType(i, item.stereoType);
                qmlMainWindow.setTempAudioSettings(i, item.audioType);
                qmlMainWindow.setForegroundSlot(i);
            }
        }

        qmlMainWindow.openTemplateVideoIniFile(getSlotListByString(), templateLoadSettingbox.getOrderList());
    }

    function globalSettings(){
        globalSettingbox.changeVideoMode();
        globalSettingbox.getGlobalValues();
        if(globalSettingbox.state == "expanded"){
            globalSettingbox.state = "collapsed";
        }else if(globalSettingbox.state == "collapsed"){
            globalSettingbox.state = "expanded";
        }
    }

    function setCameraSettings(index){
        cameraSettingsbox.setCameraValues(index);
    }

    function getCameraSettings(index)
    {
        cameraSettingsbox.getCameraValues(index);
    }

    function checkExistVideo(videoPath){
        var isExist = false;
        for(var i=0;i<videoList.count;i++){
            if(videoList.get(i).titleText === videoPath) {
                isExist = true;
                break;
            }
        }
        if(isExist) return;
        setVideoFile(videoPath)
    }

    function setVideoFiles(fileUrls_) {

        var existingVideoList = []
        var newVideoList = []
        var fileUrls = []

        for (var l = 0; l < fileUrls_.length; ++l)
        {
            videoIndex = l;
            fileUrls.push(fileUrls_[l].toString().substring(8));
        }

        for (var i=0;i<fileUrls.length;i++) {
            var fileUrl = fileUrls[i]
            for (var j=0;j<videoList.count;j++) {
                var video = videoList.get(j)

                if (fileUrl === video.titleText) {
                    existingVideoList.push(fileUrl)
                } else {
                    if (j == videoList.count - 1) {
                        newVideoList.push(fileUrl)
                    }
                }
            }
        }

        if (newVideoList.length == 0) {
            newVideoList = existingVideoList
        } else {
            newVideoList.concat(existingVideoList)
        }

        //initVideoSlots()
        for (var i = 0; i < 12; i ++) {
            videoList.set(i, {
                 "titleText": "Empty Slot",
                 "leftColor": "#ffffff",
                 "rightColor": "#8a8a8a","isSelect": false,
                 "selectStereoLeft": false,"selectStereoRight": false,
                 "selectAudioLeft": true,"selectAudioRight": false,
                 "audioType": 1,"stereoType": 0,"titleColor": "#8a8a8a",
                 "stereoLeftColor": "#8a8a8a",
                 "stereoRightColor": "#8a8a8a",
                 "footageImageName": "Background weight map",
                 "isSelectedAsBackground": false
                 });
        }

        checkTemplateMode();
        videoIndex = 0

        for (var k=0;k<newVideoList.length;k++) {
            var newVideo = newVideoList[k]
            videoIndex = k
            setVideoFile(newVideo)
        }

        getInfoStr();
        singleDlg.folder = fileUrls_[0];
        globalSettingbox.setFileUrl(fileUrls_[0]);
    }

    function updateVideoSlot()
    {
        initVideoSlots();

        var foregroundCameraCount = qmlMainWindow.getCameraCount()
        var backgroundCameraCount = qmlMainWindow.getNodalVideoCount()

        for (var i = 0; i < foregroundCameraCount + backgroundCameraCount; ++i) {

            videoIndex = i;
            videoList.set(i,{"titleColor": "#ffffff"});
            videoList.set(i,{"isSelect": true});

            var slotIndex = qmlMainWindow.getNodalSlotIndex(i)
            var isNodal = false

            if (slotIndex === i)
                isNodal = true

            if (isNodal) {
                videoList.set(i,{"titleText": qmlMainWindow.getNodalVideoFilePath(i)})
                videoList.set(i, {"footageImageName": qmlMainWindow.getNodalMaskImageFilePath(i)})
                videoList.set(i,{"isSelectedAsBackground": true})
            }

            if (!isNodal) {
                videoList.set(i,{"titleText": qmlMainWindow.getVideoPath(i)});
                videoList.set(i,{"isSelectedAsBackground": false});
            }

            setStereoSettings();
            setAudioSettings();
        }

        checkTemplateMode();
        getInfoStr();
    }

    function updateTemplateSettingUI()
    {
        var strVideoPathList = "";
        for(var i = 0 ; i < videoList.count ; i++){
            var item = videoList.get(i);
            if(!item.isSelect || item.titleText === "Empty slot")
                continue;
            strVideoPathList += (item.titleText) + ",";
        }

        var videoPathList = strVideoPathList.split(",");

        initVideoSlots();
        for (var i = 0; i < videoPathList.length; ++i)
        {
            if (videoPathList[i] == "")
                continue;
            videoIndex = i;
        }
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

    function setChangeTitle()
    {
        titleText.text = "Video Setting";
    }

    function setStereoSettings()
    {
        var stereoType =  qmlMainWindow.getTempStereoType(videoIndex);

        switch(stereoType){
        case panorama_Mono:
            videoList.set(videoIndex,{"stereoLeftColor": "#8a8a8a"})
            videoList.set(videoIndex,{"stereoRightColor": "#8a8a8a"})
            videoList.set(videoIndex,{"selectStereoLeft": false})
            videoList.set(videoIndex,{"selectStereoRight": false})
            videoList.set(videoIndex,{"stereoType": 0});
            break;
        case panorama_LeftEye:
            videoList.set(videoIndex,{"stereoLeftColor": "#ffffff"})
            videoList.set(videoIndex,{"stereoRightColor": "#8a8a8a"})
            videoList.set(videoIndex,{"selectStereoLeft": true})
            videoList.set(videoIndex,{"selectStereoRight": false})
            videoList.set(videoIndex,{"stereoType": 1});
            break;
        case panorama_RightEye:
            videoList.set(videoIndex,{"stereoLeftColor": "#8a8a8a"})
            videoList.set(videoIndex,{"stereoRightColor": "#ffffff"})
            videoList.set(videoIndex,{"selectStereoLeft": false})
            videoList.set(videoIndex,{"selectStereoRight": true})
            videoList.set(videoIndex,{"stereoType": 2});
            break;
        case panorama_BothEye:
            videoList.set(videoIndex,{"stereoLeftColor": "#ffffff"})
            videoList.set(videoIndex,{"stereoRightColor": "#ffffff"})
            videoList.set(videoIndex,{"selectStereoLeft": true})
            videoList.set(videoIndex,{"selectStereoRight": true})
            videoList.set(videoIndex,{"stereoType": 3});
            break;
        default:
            break;
        }
    }

    function setAudioSettings()
    {
        var audioType =  qmlMainWindow.getTempAudioSettings(videoIndex)
        switch(audioType){
        case mixedChannel:
            videoList.set(videoIndex,{"leftColor": "#ffffff"})
            videoList.set(videoIndex,{"rightColor": "#ffffff"})
            videoList.set(videoIndex,{"selectAudioLeft": true})
            videoList.set(videoIndex,{"selectAudioRight": true})
            videoList.set(videoIndex ,{"audioType": 0});
            break;
        case leftChannel:
            videoList.set(videoIndex,{"leftColor": "#ffffff"})
            videoList.set(videoIndex,{"rightColor": "#8a8a8a"})
            videoList.set(videoIndex,{"selectAudioLeft": true})
            videoList.set(videoIndex,{"selectAudioRight": false})
            videoList.set(videoIndex ,{"audioType": 1});
            break;
        case rightChannel:
            videoList.set(videoIndex,{"leftColor": "#8a8a8a"})
            videoList.set(videoIndex,{"rightColor": "#ffffff"})
            videoList.set(videoIndex,{"selectAudioLeft": false})
            videoList.set(videoIndex,{"selectAudioRight": true})
            videoList.set(videoIndex ,{"audioType": 2});
            break;
        case noAudio:
            videoList.set(videoIndex,{"leftColor": "#8a8a8a"})
            videoList.set(videoIndex,{"rightColor": "#8a8a8a"})
            videoList.set(videoIndex,{"selectAudioLeft": false})
            videoList.set(videoIndex,{"selectAudioRight": false})
            videoList.set(videoIndex ,{"audioType": 3});
            break;
        default:
            break;
        }
    }

	function updateTemplateLoadSettingbox()
	{
        var index = 0;
        var curSlotList = [];
        for(var i = 0 ; i < videoList.count ; i++){
            var item = videoList.get(i);
            if(!item.isSelect || item.titleText === "Empty slot")  continue;

            curSlotList[index] = item.titleText;
            index++;
        }
        templateLoadSettingbox.updateUI(2, curSlotList);
    }

    function getSlotListByString()
    {
        var strSlotList = [];
        for(var i = 0 ; i < videoList.count ; i++){
            var item = videoList.get(i);
            if(!item.isSelect || item.titleText === "Empty slot")  continue;

            strSlotList.push(item.titleText);
        }
        return strSlotList;
    }
}
