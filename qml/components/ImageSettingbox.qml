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

Item {
    id:imageSettingsBox
    width : 1030
    height: 684
    opacity: 1
    z: 3
    property int         imageIndex: -1
    property int        panorama_Mono: 0
    property int        panorama_LeftEye: 1
    property int        panorama_RightEye: 2
    property int        panorama_BothEye: 3
    property var	slotInfoList : []

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
        z: 1
        opacity: 1

        Text {
            id: titleText
            x: 30
            y: (parent.height - height) / 2
            z: 3
            color: "#ffffff"
            text: "Frame Sequence Template"
            font.bold: false
            font.pixelSize: 20
        }


        Item {
            id: groupControl
            x: root.width - groupControl.width
            width: 230
            height: 48
            z: 1
            //visible: false

            ToolbarItem {
                id: checkItem
                anchors.right: uncheckItem.left
                z: 2
                imgUrl: "../../resources/check.png"
                title: "Create Configuration"
                spacing: 4

                onClicked: {
                    var imageLock = true;
                    for(var i=0;i<imageList.count;i++){
                        if(!imageList.get(i).isSelect)  continue;
                        else {
                            imageLock = false;
                            break;
                        }

                    }

                    if(imageLock)
                        console.log("Camera is not selected")
                    else {
                        var isPrefix = false;

                        for(var i=0;i<imageList.count;i++){
                            if(imageList.get(i).prefixText === "") continue;
                            else {
                                isPrefix = true;
                                break;
                            }
                        }

                        if(!isPrefix){
                            root.showNotifyMsg("Warning","Please set the prefix");
                            return;
                        }

                        imageSettingbox.state = "collapsed";
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
                 onTriggered: openImageConfiguration()
             }

            ToolbarItem {
                id: uncheckItem
                anchors.right: plusItem.left
                z: 2
                imgUrl: "../../resources/uncheck.png"
                title: "Back"
                spacing: 4

                onClicked: {
                    slotInfoList = []
                    imageSettingbox.state = "collapsed";
                    imageTempCamSetting.state = "collapsed";
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
                spacing: 4

                onClicked: {
                    imageDialog.open();
                }
            }

            ToolbarItem  {
                id: saveItem
                anchors.right: arrangeItem.left
                z: 2
                spacing: 4
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
                title: "Select Rig Input"
                imgUrl: "../../resources/applicationSetting.png"

                onClicked: {
                    templateLoadSettingbox.x = (centralItem.width - templateLoadSettingbox.width) / 2;
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
                z: 2

                onClicked: {
                    //if(slotInfoList.length === 0) return;
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
        opacity: 1.0
    }

    ListView {
        id: imageListView
        y: 80
        width: parent.width
        height: 50 * count

        model: ListModel {
            id: imageList
        }
        delegate: Item {
            x: 5
            width: parent.width
            height: 50
            Row {
                ImageListitem {
                    title: titleText
                    titleTextColor: titleColor
                    checkSelect: isSelect
                    stereoLColor: stereoLeftColor
                    stereoRColor: stereoRightColor
                    tempStereoType: stereoType
                    isStereoLeft: selectStereoLeft
                    isStereoRight: selectStereoRight
                    prefixStr: prefixText
                }
            }
        }

    }

    ImageTempCamSetting {
        id: imageTempCamSetting
        anchors.right: parent.right
        anchors.rightMargin: 100
        width: 300
        height: 0
        z: 1
        state: "collapsed"

           states: [
               State {
                   name: "collapsed"
                   PropertyChanges { target: imageTempCamSetting; height: 0}
                   PropertyChanges { target:  imageTempCamSetting;width: 0

                   }
               },
               State {
                   name: "expanded"
                   PropertyChanges { target: imageTempCamSetting; height: 140}
                   PropertyChanges {target: imageTempCamSetting;width: 300}
               }
           ]

           transitions: [
               Transition {
                   NumberAnimation { target: imageTempCamSetting; property: "height"; duration: 300 }
                   NumberAnimation { target: imageTempCamSetting;property: "width";duration: 300}
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
        id: imageDialog
        title: "Select path of sequence files"
        selectMultiple: false
        selectFolder: true
        onSelectionAccepted: {
             var imagePath = fileUrl.toString().substring(8); // Remove "file:///" prefix
            var pathList = qmlMainWindow.onImageFileDlg(imagePath);
            var prevIdx = 0;
            var curIdx = 0;
            imageIndex = -1;
            do  {
                imageIndex++;
                curIdx = pathList.indexOf(",", prevIdx == 0 ? prevIdx : prevIdx + 1);
                if (curIdx < prevIdx || curIdx == -1) break;
                var curPath = pathList.substring(prevIdx == 0 ? prevIdx : prevIdx + 1, curIdx);
                checkExistImage(imagePath + "/" + curPath);
                prevIdx = curIdx;
            } while (curIdx !== -1);

            imageDialog.folder = imagePath;
            globalSettingbox.setFileUrl(fileUrl);
            //getAllInfoStr();
            singleImageDialog.folder = fileUrl;
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
        id: singleImageDialog
        title: "Select path of sequence files"
        //nameFilters: [ "Image files (*.jpg *.png)", "All files (*)"  ]
        selectMultiple: false
        selectFolder: true
        onSelectionAccepted: {
            var imagePath = fileUrl.toString().substring(8); // Remove "file:///" prefix
            var imageUrl = fileUrl + "/";
            imageDialog.folder = imagePath;
            setImageFile(imagePath);
            //getInfoStr();
            imageDialog.folder = fileUrl;
        }
    }

    function initImageSlots(){
        imageList.clear();
        for (var i = 0; i < 12; i ++)
        {
            imageList.append({
                  "titleText": "Empty Slot","isSelect": false,"titleColor": "#8a8a8a",
                  "stereoLeftColor": "#8a8a8a", "stereoRightColor": "#8a8a8a",
                  "stereoType": 1,
                  "selectStereoLeft": false,"selectStereoRight": false,"prefixText": ""});

        }

        checkTemplateMode();
    }

    function checkExistImage(imagePath){
        var isExist = false;
        for(var i=0;i<imageList.count;i++){
            if(imageList.get(i).titleText === imagePath) {
                isExist = true;
                break;
            }
        }

        if(isExist) return;
        setImageFile(imagePath);

    }

    function setImageFile(imagePath){
        imageList.set(imageIndex,{"titleColor": "#ffffff"});
        imageList.set(imageIndex,{"isSelect": true});
        imageList.set(imageIndex,{"titleText": imagePath});
        imageList.set(imageIndex,{"prefixText": ""});

        updateTemplateLoadSettingbox();
        setStereoSettings();
    }

    function  getInfoStr()
    {
        var imageName = imageList.get(imageIndex).titleText;
        slotInfoList = qmlMainWindow.getSlotInfo(imageName,"jpg",TemplateModes.IMAGE);
    }

    function getAllInfoStr(){
        var imageName = imageList.get(0).titleText;
        var fileExt = qmlMainWindow.getTempFileExt();
        slotInfoList = qmlMainWindow.getSlotInfo(imageName,fileExt,TemplateModes.IMAGE);

		if (slotInfoList.length === 3) {
			qmlMainWindow.setTempWidth(slotInfoList[0]);
			qmlMainWindow.setTempHeight(slotInfoList[1]);
        }
    }

	property var isNextProjectQueued: false

	function openImageConfiguration()
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
        getAllInfoStr();
        
		//root.deleteCameraViews();        
        //qmlMainWindow.disconnectProject();
        qmlMainWindow.openProject();

		root.createCameraViews();

		root.panoMode = 2;
        root.onChangePanoMode();

    }

    function setGlobalSettings(){
        qmlMainWindow.initTemplateImageObject()

		var index = 0;
        for(var i=0;i<imageList.count;i++){
            var item = imageList.get(i);
            if(!item.isSelect || item.titleText === "Empty slot")  continue;

            qmlMainWindow.sendImagePath(index, item.titleText);
            qmlMainWindow.setTempStereoType(index, item.stereoType);
			index++;
        }
		if (slotInfoList.length === 3)
		{
			qmlMainWindow.setTempWidth(slotInfoList[0]);
			qmlMainWindow.setTempHeight(slotInfoList[1]);
		}
        qmlMainWindow.openTemplateImageIniFile(getSlotListByString(), templateLoadSettingbox.getOrderList());
    }

    function globalSettings(){
        globalSettingbox.changeImageMode();
        imageTempCamSetting.state = "collapsed";
        globalSettingbox.getGlobalValues();
        if(globalSettingbox.state == "expanded"){
            globalSettingbox.state = "collapsed";
        }else if(globalSettingbox.state == "collapsed"){
            globalSettingbox.state = "expanded";
        }
    }

    function getResolution(){
        globalSettingbox.getResolution();
    }

    function setCameraSettings(index){
        imageTempCamSetting.setCameraValues(index);
    }

    function getCameraSettings(index)
    {
        imageTempCamSetting.getCameraSetting(index);
    }

    function getCameraValues(videoIndex)
    {
        var type =  qmlMainWindow.getTempStereoType(videoIndex)
        switch(type){
        case 0:
            imageList.set(videoIndex,{"stereoLeftColor": "#8a8a8a"})
            imageList.set(videoIndex,{"stereoRightColor": "#8a8a8a"})
            break;
        case 1:
            imageList.set(videoIndex,{"stereoLeftColor": "#8a8a8a"})
            imageList.set(videoIndex,{"stereoRightColor": "#8a8a8a"})
            break;
        case 2:
            imageList.set(videoIndex,{"stereoLeftColor": "#8a8a8a"})
            imageList.set(videoIndex,{"stereoRightColor": "#8a8a8a"})
            break;
        case 3:
            imageList.set(videoIndex,{"stereoLeftColor": "#8a8a8a"})
            imageList.set(videoIndex,{"stereoRightColor": "#8a8a8a"})
            break;
        default:
            break;
        }
    }

    function updateImageSlot()
    {
        initImageSlots();
        for (var i = 0; i < qmlMainWindow.getCameraCount(); ++i)
        {
            imageIndex = i;
            imageList.set(i,{"titleColor": "#ffffff"});
            imageList.set(i,{"isSelect": true});
            imageList.set(i,{"titleText": qmlMainWindow.getImagePath(i)});
            imageList.set(i,{"prefixText": qmlMainWindow.getTempImagePrefix(i)});
            setStereoSettings();
        }
        //getAllInfoStr();        

        checkTemplateMode();
    }

	function updateTemplateSettingUI()
    {
        var strImagePathList = "", strPrefixList = "";
        for(var i = 0 ; i < imageList.count ; i++){
            var item = imageList.get(i);
            if(!item.isSelect || item.titleText === "Empty slot")
                continue;
            strImagePathList += (item.titleText) + ",";
            strPrefixList += (item.prefixText) + ",";
        }

        var imagePathList = strImagePathList.split(",");
        var prefixList = strPrefixList.split(",");

        initImageSlots();
        for (var i = 0; i < imagePathList.length; ++i)
        {
            if (imagePathList[i] == "")
                continue;
            imageIndex = i;
            imageList.set(i,{"titleColor": "#ffffff"});
            imageList.set(i,{"isSelect": true});
            imageList.set(i,{"titleText": imagePathList[i]});
            imageList.set(i,{"prefixText": prefixList[i]});
            setStereoSettings();
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
        titleText.text = "Frame Sequence Setting";
    }

    function setStereoSettings()
    {
        var stereoType =  qmlMainWindow.getTempStereoType(imageIndex)
        switch(stereoType){
        case panorama_Mono:
            imageList.set(imageIndex,{"stereoLeftColor": "#8a8a8a"})
            imageList.set(imageIndex,{"stereoRightColor": "#8a8a8a"})
            imageList.set(imageIndex,{"selectStereoLeft": false})
            imageList.set(imageIndex,{"selectStereoRight": false})
            imageList.set(imageIndex,{"stereoType": 0});
            break;
        case panorama_LeftEye:
            imageList.set(imageIndex,{"stereoLeftColor": "#ffffff"})
            imageList.set(imageIndex,{"stereoRightColor": "#8a8a8a"})
            imageList.set(imageIndex,{"selectStereoLeft": true})
            imageList.set(imageIndex,{"selectStereoRight": false})
            imageList.set(imageIndex,{"stereoType": 1});
            break;
        case panorama_RightEye:
            imageList.set(imageIndex,{"stereoLeftColor": "#8a8a8a"})
            imageList.set(imageIndex,{"stereoRightColor": "#ffffff"})
            imageList.set(imageIndex,{"selectStereoLeft": false})
            imageList.set(imageIndex,{"selectStereoRight": true})
            imageList.set(imageIndex,{"stereoType": 2});
            break;
        case panorama_BothEye:
            imageList.set(imageIndex,{"stereoLeftColor": "#ffffff"})
            imageList.set(imageIndex,{"stereoRightColor": "#ffffff"})
            imageList.set(imageIndex,{"selectStereoLeft": true})
            imageList.set(imageIndex,{"selectStereoRight": true})
            imageList.set(imageIndex,{"stereoType": 3});
            break;
        default:
            break;
        }
    }

    function updateTemplateLoadSettingbox()
    {
        var index = 0;
		var curSlotList = [];
        for(var i =  0; i < imageList.count; i++){
            var  item = imageList.get(i);
            if(item.isSelect) {
				curSlotList[index] = item.titleText;
				index++;
			}                
        }
        templateLoadSettingbox.updateUI(3, curSlotList);
    }

	function getSlotListByString()
    {
        var strSlotList = [];
        for(var i =  0; i < imageList.count; i++){
            var item = imageList.get(i);
            if(!item.isSelect || item.titleText === "Empty slot")  continue;

            strSlotList.push(item.titleText);
        }
        return strSlotList;
    }
}
