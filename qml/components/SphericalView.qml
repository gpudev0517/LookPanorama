import QtQuick 2.5
import	MCQmlCameraView 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.2
import QtQuick.Dialogs 1.2

Item {
    id: spherical
    width: centralItem.width
    height:　centralItem.height
    visible: false
    property int    cellWidthVal: 425
    property int    cellHeightVal: 425
    property var    seamLabelList: []
    property var    stitchviewList : null
    property bool   isSeam: false
    property int	selectedSeamIndex1: -1;
    property int    selectedSeamIndex2: -1;
    property bool   isBanner: false
    property bool   bannerEnded: false
    property var    clickPos
    property var    releasedPos
    property var    movedPos: "0,0"
    property var    curPosX
    property var    curPosY
    property var    prePosX
    property var    prePosY
    property var    ctxList: []
    property int    adIndex: -1
    property var    adPoints: []
    property int    delta: 20
    property bool   isEndPos: false
    property int    bannerIndex: -1
    property var    bannerPointList: []

    property bool   isWeightMap: false

	Component.onCompleted: {
		for (var i = 0; i < 16; i++)
		{
			var component = Qt.createComponent("SeamLabel.qml");
			if (component.status === Component.Ready)
			{
				var seamLabel = component.createObject(spherical, {"x": -1, "y": -1,"z": 10,"seamIndex": i + 1, "visible": false});
				seamLabelList[i] = seamLabel;
			}
		}
		weightmapSettingbox.cameraParamsSettingbox = cameraParamsSettingbox;
	}

    Component.onDestruction: {
		for( var i = 0; i < 16; i++ )
		{
			var seamLabel = seamLabelList[i];
			if (typeof(seamLabel.destroy) == "function")
				seamLabel.destroy();
		}
    }

    onWidthChanged: {
        if(isBanner ){
            getPosPano2Window();
        }
    }

    onHeightChanged: {
        if(isBanner){
            getPosPano2Window();
        }
    }

    Rectangle {
        id: backgroundRectangle
        color: "#000000"
        width: parent.width
        height: parent.height
        GridView {
            id: sphericalGridView
            anchors.fill: parent

			MCStitchCameraView {
				id: sphericalObject
				x: 0
				y: 0
			}
        }
    }

    WeightmapSettingbox{
           id: weightmapSettingbox
           z:  -1
           x: root.width
           visible:  false		   
    }
    
	CameraParamsSettingbox{
           id: cameraParamsSettingbox
           z:  -1
           x: root.width
           visible:  false
    }

    RigLoadSettingbox{
           id: rigLoadSettingbox
           z:  10
           x: root.width * 0.8
           visible:  false
    }

    RigSaveSettingbox{
           id: rigSaveSettingbox
           z:  10
           x: root.width * 0.8
           visible:  false
    }

	CTSettingbox {
			id: ctSettingbox
			x: root.width
			z: 10
			visible: false
     }

	 DisplayLUTControl
     {
        id: lutControl
        visible: false
     }

	 StitchCameraTemplateSettingbox {
			id: stitchCameraTemplateSettingbox
			x: root.width
			z: 10
			visible: false
     }

    Rectangle {
        width: parent.width
        height: parent.height
        opacity: 0.0
        z: 1

        Timer {
            id: weightmapTimer
            interval: 30
            running: false
            repeat: true
            onTriggered: {
                qmlMainWindow.drawWeightMap(centralItem.width, centralItem.height, movedPos.x, movedPos.y);
            }

        }

        MouseArea {
            id: mouseArea
            anchors.fill: parent
            hoverEnabled: true
            onPressed: {
                clickPos  = mapToItem(parent, mouse.x,mouse.y);
                if(!isBanner){
					if(!statusItem.isRecording)
					{
						qmlMainWindow.onPressedSpherical(clickPos.x, clickPos.y);
					}
                } else {
                    prePosX = clickPos.x;
                    prePosY = clickPos.y;
                    curPosX = clickPos.x;
                    curPosY = clickPos.y;
                    if(adIndex > 3){
                        return;
                    }

                    adIndex++;

                    adPoints[adIndex] = clickPos;
                }

				if (weightmapSettingbox.isEditweightmap)
				{
					if(pressedButtons != Qt.LeftButton) return;
                    //qmlMainWindow.drawWeightMap(centralItem.width, centralItem.height, movedPos.x, movedPos.y);
                    weightmapTimer.restart();
				}
            }


            onReleased: {
                releasedPos = mapToItem(parent, mouse.x,mouse.y);
                if (!isBanner){
					if(!statusItem.isRecording)
					{
						qmlMainWindow.onReleasedSpherical(releasedPos.x, releasedPos.y);
					}
                }else{
                    if(adIndex > 3) return;
                    curPosX = releasedPos.x;
                    curPosY = releasedPos.y;
                    getPosWindow2Pano(sphericalView.width,sphericalView.height,curPosX,curPosY);

                    if(adIndex === 3){
                        bannerEnded = true;
                        bannerFileDialog.open();
                    }
                    createBannerPoint(curPosX,curPosY,adIndex);
                }

                if (weightmapSettingbox.isEditweightmap && mouse.button == Qt.LeftButton)
				{
                    qmlMainWindow.setWeightMapChanged(false);
                    weightmapTimer.stop();
				}
            }

            onPositionChanged: {
                movedPos = mapToItem(parent,mouse.x,mouse.y);
                cursorShape = toolbox.isWeightmap ? weightmapSettingbox.currentCursorShape: Qt.ArrowCursor;
				
				if(isSeam){
                    updateSeam()
                    if (weightmapSettingbox.isMirror)
                        updateSeam_()
                }

                if(!isBanner && !weightmapSettingbox.isEditweightmap){
					if(!statusItem.isRecording)
					{
						qmlMainWindow.onMovedSpherical(movedPos.x, movedPos.y);
						return;
					}
                }

                if (weightmapSettingbox.isEditweightmap && mouse.button == Qt.LeftButton)
				{
                    qmlMainWindow.drawWeightMap(centralItem.width, centralItem.height, movedPos.x, movedPos.y);
                    weightmapTimer.restart();
				}
            }

            acceptedButtons: isBanner ? Qt.LeftButton | Qt.RightButton: Qt.LeftButton

			onDoubleClicked: {
				movedPos  = mapToItem(parent,mouse.x,mouse.y);				
				qmlMainWindow.onDoubleClickedSpherical(movedPos.x, movedPos.y);
			}

            onClicked: {
                if (mouse.button !== Qt.RightButton) return;

                if (!isBanner) return;
                clearAllBannerPoint();

                if(bannerIndex == -1) return;
                qmlMainWindow.removeLastBanner();
                bannerCtrl.removeLastBannerList();

                bannerCtrl.updateAllBannerList();

                if(bannerCtrl.bannerListCount() === 0){
                    bannerCtrl.initBannerList();
                    return;
                }
            }

        }
    }

    FileDialog {
        id: bannerFileDialog
        title: "Open Banner image for video file"
        nameFilters: [ "Banner file (*.mp4 *.avi *.jpg *.png *.bmp)", "All files (*)" ]
        selectMultiple: true
        onSelectionAccepted: {
            clearAllBannerPoint();
            var isVideo = true;
            var bannerPath = fileUrl.toString().substring(8); // Remove "file:///" prefix
            var pos = bannerPath.lastIndexOf(".") + 1;
            var fileExt = bannerPath.substring(pos);
            if(fileExt === "mp4" || fileExt === "avi"){
                isVideo = true;
            }
            else if( fileExt === "jpg" || fileExt === "png" || fileExt === "bmp"){
                isVideo = false;
            }

            qmlMainWindow.addBanner(centralItem.width, centralItem.height,
				adPoints[0],adPoints[1], adPoints[2],adPoints[3],
				bannerPath,isVideo);

            bannerEnded = false;
            bannerIndex++;
            bannerCtrl.insertBannerList((bannerCtrl.bannerListCount() - 1) + 1);
        }
    }

    function getPosWindow2Pano(width,height,x,y){

        bannerPointList[adIndex] = qmlMainWindow.getPosWindow2Pano(width,height,x,y);
    }

    function getPosPano2Window(){
        var wPos;
        var wPosList;
        for(var i = 0; i < adIndex + 1; i++){
            var pos = bannerPointList[i];
            var posList = pos.split(":");
            wPos = qmlMainWindow.getPosPano2Window(sphericalView.width,sphericalView.height,posList[0],posList[1]);
            wPosList = wPos.split(":");
            clearBannerPoint(i);
            createBannerPoint(wPosList[0],wPosList[1],i);
            adPoints[i].x = wPosList[0];
            adPoints[i].y = wPosList[1];
        }

    }

    function openBannerFileDialog(){
        bannerFileDialog.open();
    }

	function openCTSettingbox() {
		ctSettingbox.visible = true;
		toolbox.isCTSetting = true;

		ctSettingbox.x = (centralItem.width - ctSettingbox.width) / 2 ;
        ctSettingbox.y = 0;        

		ctSettingbox.setColorTemperature(qmlMainWindow.getColorTemperature());
	}

	function closeCTSettingbox() {
		ctSettingbox.visible = false;
		toolbox.isCTSetting = false;
	}

	function onOpenStitchCameraTemplateSettingbox() {
		stitchCameraTemplateSettingbox.visible = true;
		toolbox.isStitchCameraTemplate = true;		

		stitchCameraTemplateSettingbox.x = centralItem.width - stitchCameraTemplateSettingbox.width;
        stitchCameraTemplateSettingbox.y = 0;

		stitchCameraTemplateSettingbox.initCameraParms();
	}

	function onCloseStitchCameraTemplateSettingbox() {
		stitchCameraTemplateSettingbox.visible = false;
		toolbox.isStitchCameraTemplate = false;		
	}

	function reloadStitchCameraTemplateParameters() {
		stitchCameraTemplateSettingbox.initCameraParms();		
	}

	function reverseWeightMapViewModeSwitchStatus() {
		return weightmapSettingbox.reverseViewModeSwitchStatus();
	}

    function createWeightmapSettingbox(){
        weightmapSettingbox.visible = true;
        weightmapSettingbox.isEditweightmap = true;
        updateWeightmapSettingsbox();
        weightmapSettingbox.appendCameraCombo();
        weightmapSettingbox.setDrawWeightmapSettings();

		cameraParamsSettingbox.getCameraParams(cameraParamsSettingbox.m_curCameraIndex);
		cameraParamsSettingbox.visible = true;	
        weightmapSettingbox.initWeightmapSettings();
    }

    function updateWeightmapSettingsbox(){
		cameraParamsSettingbox.x = centralItem.width - cameraParamsSettingbox.width;
        cameraParamsSettingbox.y = 0;
        cameraParamsSettingbox.z = 10;

        weightmapSettingbox.x = centralItem.width - weightmapSettingbox.width - cameraParamsSettingbox.width - delta;
        weightmapSettingbox.y = 0;
        weightmapSettingbox.z = 10;		
    }

    function initWeightmapSettingbox(){
        closeWeightmapSettingbox();
        weightmapSettingbox.initWeightmapSettings();
    }

    function closeWeightmapSettingbox(){
        weightmapSettingbox.isEditweightmap = false;
        weightmapSettingbox.visible = false;
        weightmapSettingbox.currentCursorShape = Qt.CrossCursor;

		cameraParamsSettingbox.visible = false;

        closeSeam();
    }

	function closeSeam()
	{
        qmlMainWindow.enableSeam(-1, -10);
		isSeam = false
		hideSeamLabels()
	}

    function updateCameraIndex(index) {
        weightmapSettingbox.updateCameraIndex(index);
    }

    function setUndoStatus(weightMapEditUndoStatus) {
        weightmapSettingbox.setUndoStatus(weightMapEditUndoStatus);
    }

    function setRedoStatus(weightMapEditRedoStatus) {
        weightmapSettingbox.setRedoStatusRedo(weightMapEditRedoStatus);
    }

	function resetWeightMapSettingView() {		
		weightmapSettingbox.getCameraParams(-1);
	}

	function updateConfiguration()
	{
		qmlMainWindow.updateStitchView(sphericalObject.camView)
	}

	function updatePlaybackConfiguration()
	{
		qmlMainWindow.updatePlaybackView(sphericalObject.camView)
	}

    function createScreenNumber()
    {
        var component = Qt.createComponent("ScreenNumber.qml");
        if (component.status === Component.Ready) {
            var screenNum = component.createObject(spherical, {"x": 0, "y": 0});
        }
        else
            console.log(component.errorString());
    }

	function updateFullscreen(screenNum)
	{
		qmlMainWindow.updateFullScreenStitchView(fullscreenObject.camView, screenNum - 1);
	}

	property int count: 0

	function updateSeam()
	{
		//console.log("updateSeam1");
		if(!isSeam) return;
		var cameraCnt = qmlMainWindow.getCameraCount();
		//console.log("updateSeam2");



        if(selectedSeamIndex1 === 0)
        {
            for (var i = 0; i < cameraCnt; i++)
            {
                var seamLabelPos = qmlMainWindow.getSeamLabelPos(i);
                // left eye
                if ((qmlMainWindow.isStereo() && qmlMainWindow.isLeftEye(i)) ||
                    qmlMainWindow.isStereo() === false)
                    showSeamLabel(seamLabelPos, i, false);
                // right eye
                if (qmlMainWindow.isStereo()  && qmlMainWindow.isRightEye(i))
                    showSeamLabel(seamLabelPos, i, true);
            }
        }
        else
        {
			hideSeamLabels();
            var i = selectedSeamIndex1 - 1;
            var seamLabelPos = qmlMainWindow.getSeamLabelPos(i);
            // left eye
            if ((qmlMainWindow.isStereo() && qmlMainWindow.isLeftEye(i)) ||
                qmlMainWindow.isStereo() === false)
                showSeamLabel(seamLabelPos, i, false);
            // right eye
            if (qmlMainWindow.isStereo()  && qmlMainWindow.isRightEye(i))
                showSeamLabel(seamLabelPos, i, true);
        }
	}

    function updateSeam_()
    {
        //console.log("updateSeam1");
        if(!isSeam) return;
        var cameraCnt = qmlMainWindow.getCameraCount();
        //console.log("updateSeam2");

       

        if(selectedSeamIndex2 === 0)
        {
            for (var i = 0; i < cameraCnt; i++)
            {
                var seamLabelPos = qmlMainWindow.getSeamLabelPos(i);
                // left eye
                if ((qmlMainWindow.isStereo() && qmlMainWindow.isLeftEye(i)) ||
                    qmlMainWindow.isStereo() === false)
                    showSeamLabel(seamLabelPos, i, false);
                // right eye
                if (qmlMainWindow.isStereo()  && qmlMainWindow.isRightEye(i))
                    showSeamLabel(seamLabelPos, i, true);
            }
        }
        else
        {
            //hideSeamLabels();
            if (!weightmapSettingbox.isMirror) {
                hideSeamLabels()
            }

            var i = selectedSeamIndex2 - 1;
            var seamLabelPos = qmlMainWindow.getSeamLabelPos(i);
            // left eye
            if ((qmlMainWindow.isStereo() && qmlMainWindow.isLeftEye(i)) ||
                qmlMainWindow.isStereo() === false)
                showSeamLabel(seamLabelPos, i, false);
            // right eye
            if (qmlMainWindow.isStereo()  && qmlMainWindow.isRightEye(i))
                showSeamLabel(seamLabelPos, i, true);
        }
    }

    function setSeamIndex(cameraIndex1, cameraIndex2)
	{
        selectedSeamIndex1 = cameraIndex1
        updateSeam()

        if (weightmapSettingbox.isMirror) {           
            selectedSeamIndex2 = cameraIndex2
            updateSeam_()
        }
	}

    function showSeamLabel(seamPos, index, isRight)
    {
		var isStereo = qmlMainWindow.isStereo();

        var frameWidth = centralItem.width;
        var frameHeight = centralItem.height;

        var panoWidth = qmlMainWindow.getTempPanoWidth();
		var panoH = qmlMainWindow.getTempPanoHeight();
        var panoHeight = panoH;		
		if (isStereo)
			panoHeight *= 2;

        var curWidth = frameWidth;
        var curHeight = frameHeight;

        if(panoHeight / panoWidth > frameHeight / frameWidth)
        {
            curHeight = centralItem.height;
            curWidth = curHeight * (panoWidth / panoHeight);
        }
        else
        {
            curWidth = centralItem.width;
            curHeight = curWidth * panoHeight / panoWidth;
        }
		var xOrg = (frameWidth - curWidth) / 2;
		var yOrg = (frameHeight - curHeight) / 2;
        
        var xPos = (curWidth / panoWidth) * seamPos[0] + xOrg;
        var yPos = (curHeight / panoHeight) * seamPos[1] + yOrg;
		if (isStereo && isRight)
			yPos += curHeight/2;

		var labelIndex = isRight * 8 + index;
		var seamLabel = seamLabelList[labelIndex];
		seamLabel.visible = true;
		seamLabel.x = xPos;
		seamLabel.y = yPos;
        seamLabel.seamIndex = index + 1;
    }

    function  hideSeamLabels()
    {
        for (var i = 0; i < 16; i++)
        {
            var seamLabel = seamLabelList[i];
			seamLabel.visible = false;
        }
    }

    function clearStitchViews()
    {
        if(stitchviewList == null) return;
        for (var i = 0; i < stitchviewList.length; i++)
        {
            var component = stitchviewList[i];
            if (typeof(component.destroy) == "function")
                component.destroy();
        }
    }

    function drawLine(startX,startY,endX,endY)
    {
        var component = Qt.createComponent("Line.qml");
        if (component.status === Component.Ready)
        {
            var line = component.createObject(spherical, {"prePosX": startX, "prePosY": startY, "curPosX": endX,"curPosY": endY});
            ctxList[adIndex] = line;
        }
    }

    function createBannerPoint(bannerX,bannerY,adIndex)
    {
        var component = Qt.createComponent("Line.qml");
        if (component.status === Component.Ready)
        {
            var bannerPoint = component.createObject(spherical, {"xPos": bannerX, "yPos": bannerY});
            ctxList[adIndex] = bannerPoint;
        }
    }

    function clearBannerPoint(index)
    {
        if(index === -1) return;
        var component;
        if (ctxList.length > 0 && typeof(ctxList[index]) != "undefined" ) {
            //console.log("clearIndex->" + index + " " + typeof(ctxList[index].destroy));
            component = ctxList[index];
            if (typeof(component.destroy) == "function")
                component.destroy();
        }
    }

    function clearAllBannerPoint(){
        if(adIndex === -1) return;
        for(var i = 0; i < adIndex+ 1; i ++){
            //clearLine(i);
            clearBannerPoint(i);
        }
        adIndex = -1;
    }

    function updateBannerPoint(){


    }

    function clearLine(index){
        if(index === -1) return;
        var component;
        if (ctxList.length > 0 && typeof(ctxList[index]) != "undefined" ) {
            //console.log("clearIndex->" + index + " " + typeof(ctxList[index].destroy));
            component = ctxList[index];
            if (typeof(component.destroy) == "function")
                component.destroy();
        }
    }

    function createNumberTab(){
        var numCount = qmlMainWindow.getScreenCount();
        for(var i = 0; i < numCount; i++){
            screenNumber.addTab(i + 1);
        }

        screenNumber.width = 40 * (numCount - 1);
        screenNumber.focus = true;
     }

    function initBanners(){
        bannerCtrl.state = "collapsed";
        clearAllBannerPoint();
        isBanner = false;
        bannerEnded = false;
    }

    function showLutCtrl() {
        lutControl.visible = true;
    }

	function closeLutCtrl() {
		lutControl.visible = false;
	}

	function loadLutData(lutData, colorType) {
        lutControl.onLoadData(lutData, colorType);
    }
}

