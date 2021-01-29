import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.1
import QtQuick.Dialogs 1.2
import "../controls"

Item {
    width : 350
    height: 684


    property int        hoveredType: 0
    property bool       isHovered : false
    property bool       isSelected: false
    property int        leftMargin: 20
    property int        rightMargin: 20
    property int        topMargin: 20
    property int        spacing: 20
    property int        nItemCount: 12
    property color      textColor: "#ffffff"
    property color      comboTextColor: "#7899ff"
    property color      comboBackColor: "#343434"
    property int        itemHeight:30
    property int        lblFont: 14
    property int        groupFont: 16
    property var        cameraCnt
    property bool       isBlend: true
    property color      spliterColor: "#555555"
    property int        lblWidth: 80
    property int        textWidth: 45
    property string     seamLabelPos
    property bool       isExposure: false
    property int        cropVal
    property bool       isClickedAutoCalibrate: false

	Component.onCompleted:
	{
		switchPreview();
	}

	MouseArea {
		anchors.fill: parent
		onClicked: {
		}
	}

    Rectangle {
        id: titleRectangle
        x: 0
        y: 0
        width: parent.width
        height: 48
        color: "#171717"
        z: 1

        Text {
            id: titleText
            x: (350 - width) / 2
            y: (parent.height - height) / 2
            z: 3
            color: "#ffffff"
            text: qsTr("Spherical")
            font.bold: false
            font.pixelSize: 20
        }
    }

//    Rectangle {
//        id: spliterRectangle
//        width: parent.width
//        height: 2
//        z: 3
//        anchors.top: titleRectangle.bottom
//        color: "#1f1f1f"
//    }
    Spliter {
        id: spliterRectangle
        width: parent.width
        z: 3
        anchors.top: titleRectangle.bottom
    }

    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#171717"
        opacity: 0.9
    }

    Rectangle {
        id: okRectangle
        width: parent.width * 0.5
        height: 40
        color: "#0e3e64"
        z: 1
        anchors {
            left: parent.left
            bottom: parent.bottom
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                statusItem.setPlayAndPause();
                collapseSphericalbox();

                toolbox.clearSelected();
                statusItem.setPlayAndPause();
                toolbox.initSphericalTopControls();
                exposureGroup.state = "collapsed";

            }
        }

        Image {
            id: okImage
            x: (175 - width) / 2
            y: (parent.height - height) / 2
            width: 25
            height: 25
            fillMode: Image.PreserveAspectFit
            source: "../../resources/btn_ok.PNG"
        }
    }

    Rectangle {
        id: cancelRectangle
        width: parent.width * 0.5
        height: 40
        color: "#1f1f1f"
        z: 1
        anchors {
            right: parent.right
            bottom: parent.bottom
        }
        MouseArea {
            anchors.fill: parent
            onClicked: {
                if (isClickedAutoCalibrate) {
                    statusItem.setPlayAndPause();
                    qmlMainWindow.onCancelCameraSettings();
                    qmlMainWindow.reStitch(true);
                    collapseSphericalbox();

                    statusItem.setPlayAndPause();
                    qmlMainWindow.onRollbackGain();
                }
                 toolbox.initSphericalTopControls();
                toolbox.clearSelected();
                sphericalBox.state = "collapsed";
            }
        }

        Image {
            id: cancelImage
            x: (175 - width) / 2
            y: (parent.height - height) / 2
            width: 25
            height: 25
            fillMode: Image.PreserveAspectFit
            source: "../../resources/btn_cancel.PNG"
        }
    }
    ScrollView {
        id: scrollView
        y: titleRectangle.height
        width: parent.width
        height: parent.height - titleRectangle.height - okRectangle.height
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
            height:  (spacing + itemHeight )* nItemCount + previewItem.height

            Item {
                id: cameraItem
                width: 350
                height: itemHeight
                anchors.topMargin: 20
                anchors.top: parent.top
               Text {
                   id: cameraLabel
                   anchors.left: parent.left
                   anchors.leftMargin: leftMargin
                   anchors.verticalCenter: parent.verticalCenter
                   text: qsTr("Camera")
                   color: "#ffffff"
                   font.pixelSize: lblFont

               }

               ComboBox {
                   id: cameraCombo
                   anchors.left: cameraLabel.right
                   anchors.leftMargin: leftMargin
                   anchors.verticalCenter: parent.verticalCenter
                   width:parent.width / 2
                   height: 30
                   model: ListModel {
                       id: cameraModel
                   }

                   onCurrentTextChanged:
				   {
						switchPreview();
						getExposureSetting();
						blendSettings.getBlendSettings();

						//if(!isExposure) return;
						if(seamSwitch.checked === true || isExposure === true){
							cameraList.currentIndex = currentIndex + 1;
							showSeam();
						}
                   }
               }
            }

            ListView {
                id: previewItem
                x: 30
                anchors.top: cameraItem.bottom
                anchors.topMargin: 12
                width: 300
                height: 250

				CameraPreview {
					id: cameraPreviewObject
					x: 0
					y: 0
				}
				
            }

            Item {
                id: resetCalibItem
                width: parent.width / 3
                anchors.bottom:  blendSettingGroup.top
                anchors.bottomMargin: 20
                height: 30
                anchors.left: parent.left
                anchors.leftMargin: leftMargin


                Rectangle {
                    id: resetCalibHoverRect
                    x: 0
                    width: parent.width
                    height: parent.height
                    anchors.fill: parent
                    color: "#373737"
                    visible: false
                    border.color: "#4e8d15"
                    border.width: 1
                    z: 1
                }


                Text {
                    color: "#ffffff"
                    text: "Reset calibration"
                    z: 1
                    font.pixelSize: 14
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent


                }

                Rectangle {
                    id: resetCalibRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id: resetCalibMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        onEntered: resetCalibHoverRect.visible = true;
                        onExited: resetCalibHoverRect.visible = false;

                        onClicked: {
                            qmlMainWindow.reloadCameraCalibrationFile(null);
                        }
                    }
                }
            }

                Item {
                    id: blendSettingGroup
                    anchors.top: previewItem.bottom
                    anchors.topMargin: 80
                    width: 100
                    height: 30
                    Text {
                        color: "#ffffff"
                        text: qsTr("Blend")
                        font.bold: true
                        anchors.leftMargin: 20
                        font.pixelSize: groupFont
                        anchors.left: parent.left
                    }
                }

                Spliter {
                    id: blendSettingSpliter
                    width:  parent.width - leftMargin * 3
                    anchors.top: blendSettingGroup.bottom
                    //anchors.topMargin: 20
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                }

                Item {
                    id: arrowDown_blendItem

                    anchors.left: blendSettingSpliter.right
                    anchors.leftMargin: -5
                    anchors.top: blendSettingGroup.bottom
                    anchors.topMargin: -10
                    width: 30
                    height: 30
                    z: 2

                    Image {
                        id: arrowDown_blendImage
                        x: (parent.width - width) / 2
                        y: (parent.height - height) / 2
                        z: 2
                        width: 30
                        height: 30
                        fillMode: Image.PreserveAspectFit
                        source: "../../resources/harrow_down.png"
                        visible: true
                    }

                    Image {
                        id: harrowDown_blendImage
                        x: (parent.width - width) / 2
                        y: (parent.height - height) / 2
                        z: 2
                        width: 30
                        height: 30
                        fillMode: Image.PreserveAspectFit
                        source: "../../resources/arrow_down.png"
                        visible: false
                    }
                    MouseArea {
                        z: 2
                        anchors.fill: parent
                        hoverEnabled: true
                        onEntered: {
                            arrowDown_blendImage.visible = false;
                            harrowDown_blendImage.visible = true;
                        }
                        onExited: {
                            arrowDown_blendImage.visible = true;
                            harrowDown_blendImage.visible = false;
                        }
                        onClicked:
                        {
                            arrowUp_blendItem.visible = true;
                            arrowDown_blendItem.visible = false;
                            blendSettings.state = "expanded";
                            nItemCount = 30;

                        }
                    }
                }

                Item {
                    id: arrowUp_blendItem
                    anchors.left: blendSettingSpliter.right
                    anchors.leftMargin: -1
                    anchors.top: blendSettingGroup.bottom
                    anchors.topMargin: -15
                    width: 30
                    height: 30
                    z: 2
                    visible: false

                    Image {
                        id: arrowUp_blendImage
                        x: (parent.width - width) / 2
                        y: (parent.height - height) / 2
                        z: 2
                        width: 30
                        height: 30
                        fillMode: Image.PreserveAspectFit
                        source: "../../resources/harrow_up.png"
                        visible: true
                    }

                    Image {
                        id: harrowUp_blendImage
                        x: (parent.width - width) / 2
                        y: (parent.height - height) / 2
                        z: 2
                        width: 30
                        height: 30
                        fillMode: Image.PreserveAspectFit
                        source: "../../resources/arrow_up.png"
                        visible: false
                    }
                    MouseArea {
                        z: 2
                        anchors.fill: parent
                        hoverEnabled: true
                        onEntered: {
                            arrowUp_blendImage.visible = false;
                            harrowUp_blendImage.visible = true;
                        }
                        onExited: {
                            arrowUp_blendImage.visible = true;
                            harrowUp_blendImage.visible = false;
                        }
                        onClicked:
                        {
                            arrowUp_blendItem.visible = false;
                            arrowDown_blendItem.visible = true;
                            blendSettings.state = "collapsed";
                            nItemCount = 15;

                        }
                    }
                }

                BlendSettings {
                    id: blendSettings
                    anchors.top: arrowDown_blendItem.bottom
                    anchors.left: parent.left
                    state: "collapsed"

                    states: [
                        State {
                            name: "expanded"
                            PropertyChanges {
                                target: blendSettings
                                height: 400
                                //opacityVal: 0.9
                            }
                        },
                        State {
                            name: "collapsed"
                            PropertyChanges {
                                target: blendSettings
                                height: 0
                                //opacityVal: 0.0
                            }
                        }
                    ]

                    transitions: [
                        Transition {
                            NumberAnimation { target: blendSettings; property: "height"; duration: 300 }
                            NumberAnimation { target: blendSettings; property: "opacityVal"; duration: 300 }
                        }
                    ]
                }


            Item {
                id: applyItem
                width: 70
                height: 30
                anchors.top: exposure.bottom
                anchors.left: parent.left
                anchors.leftMargin: leftMargin

                Rectangle {
                    id: applyHoverRect
                    width: parent.width
                    height: parent.height
                    color: "#373737"
                    visible: false
                    border.color: "#4e8d15"
                    z: 1

                    Text {
                        color: "#ffffff"
                        text: "Apply"
                        font.pointSize: 11
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        anchors.fill: parent
                    }
                }

                Rectangle {
                    id: applyRect
                    width: parent.width
                    height: parent.height
                    color: "#373737"

                    Text {
                        id: applyText
                        color: "#ffffff"
                        text: "Apply"
                        font.pointSize: 11
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        anchors.fill: parent


                    }

                    MouseArea {
                        id: applyMouseArea
                        anchors.fill: parent
                        hoverEnabled: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                applyHoverRect.visible = true;
                            }else{
                                applyHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            if(isExposure)
                            {
                                qmlMainWindow.setCameraExposure(cameraCombo.currentIndex, exposureText.text);
                                return;
                            }
                            //setBlendValues();
                        }
                    }
                }
            }            


            Item {
                id: exposure
                width: parent.width
                height: 130
                anchors.top: previewItem.bottom
                anchors.topMargin: 30
                visible: false
                 Item {
                    id: exposoureGroup
                    width: 100
                    height: 30
                    //anchors.topMargin: 10
                    //anchors.top: levelItem.bottom
                    Text {
                        color: "#ffffff"
                        text: qsTr("Exposure (EV Offset)")
                        font.bold: true
                        anchors.leftMargin: 20
                        font.pixelSize: groupFont
                        anchors.left: parent.left
                    }
                }

                 Spliter {
                     id: exposureSpliter
                     //height: 2
                     width: 340
                     anchors.top:  exposoureGroup.bottom
                     anchors.left: parent.left
                     anchors.leftMargin: 3
                 }

                Item {
                    id: exposureItem
                    width: 350
                    height: itemHeight
                    anchors.top: exposureSpliter.bottom
                    anchors.topMargin: spacing

                    Slider {
                        id: exposureSlider
                        value: exposureText.text
                        width: parent.width * 0.5
                        minimumValue: -2.5
                        maximumValue: 2.5
                        stepSize: 0.01
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                        anchors.right: exposureText.left
                        anchors.rightMargin: rightMargin
                        anchors.verticalCenter: parent.verticalCenter

                        onPressedChanged: {
                            if(!pressed){
                                qmlMainWindow.setCameraExposure(cameraCombo.currentIndex, exposureText.text);
                            }
                        }

                    }

                    FlatText {
                        id: exposureText
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                        width: parent.width * 0.15
                        text:  exposureSlider.value
                        maximumLength: 6
//                        onTextChanged: {
//                            exposureSlider.value = exposureText.text;
//                        }

                        onEditingFinished: {
                            exposureSlider.value = exposureText.text;
                            setExposureSetting();
                        }

                    }

                }

                Item {
                    id: resetItem
                    width: 65
                    anchors.top: exposure.bottom
                    height: 30
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin

                    Rectangle {
                        id: resetHoverRect
                        x: 0
                        width: parent.width
                        height: parent.height
                        anchors.fill: parent
                        color: "#373737"
                        visible: false
                        border.color: "#4e8d15"
                        border.width: 1
                        z: 1
                    }


                    Text {
                        id: resetText
                        z: 1
                        color: "#ffffff"
                        text: "Reset All"
                        font.pointSize: 11
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        anchors.fill: parent
                    }

                    Rectangle {
                        id: resetRect
                        width: parent.width
                        height: parent.height
                       anchors.fill: parent
                        color: "#373737"

                        MouseArea {
                            id: resetMouseArea
                            width: 60
                            anchors.fill: parent
                            hoverEnabled: true
                            onHoveredChanged: {
                                isHovered = !isHovered
                                if(isHovered){
                                    resetHoverRect.visible = true;
                                }else{
                                    resetHoverRect.visible = false;
                                }
                            }

                            onClicked: {
                                qmlMainWindow.onResetGain();
                                getExposureSetting();
                            }
                        }
                    }
                }

                Item {
                id: caliExposureItem
                width: 130
                anchors.top: exposure.bottom
                height: 30
                anchors.right: parent.right
                anchors.rightMargin: rightMargin

                    Rectangle {
                        id: caliExposureHoverRect
                        x: 0
                        width: parent.width
                        height: parent.height
                        anchors.fill: parent
                        color: "#373737"
                        visible: false
                        border.color: "#4e8d15"
                        border.width: 1
                        z: 1
                    }


                     Text {
                         id: caliExposureText
                         z: 1
                         color: "#ffffff"
                         text: "Calibrate Exposure"
                         font.pointSize: 11
                         horizontalAlignment: Text.AlignHCenter
                         verticalAlignment: Text.AlignVCenter
                         anchors.fill: parent
                     }

                      Rectangle {
                          id: caliExposureRect
                          width: parent.width
                          height: parent.height
                          anchors.fill: parent
                          color: "#373737"

                          MouseArea {
                              id: caliExposureMouseArea
                              width: 60
                              anchors.fill: parent
                              hoverEnabled: true
                              onHoveredChanged: {
                                  isHovered = !isHovered
                                  if(isHovered){
                                      caliExposureHoverRect.visible = true;
                                  }else{
                                      caliExposureHoverRect.visible = false;
                                  }
                              }

                              onClicked: {
                                  statusItem.setPauseMode();
                                  qmlMainWindow.onCalculatorGain();
                                  isClickedAutoCalibrate = true;
                              }
                        }
                    }
                }
            }

            Item {
                id: seam
                width: parent.width
                height: 200
                anchors.top: blendSettings.bottom
                anchors.topMargin: 30


                Item {
                    id: seamGroup
                    width: 100
                    height: 30
                    Text {
                        color: "#ffffff"
                        text: qsTr("Seam")
                        font.bold: true
                        anchors.leftMargin: 20
                        font.pixelSize: groupFont
                        anchors.left: parent.left
                    }
                }

                Spliter {
                    id: seamSpliter
                    x: 20
                    width: parent.width - 45
                    anchors.top:  seamGroup.bottom
                    anchors.rightMargin: rightMargin
                }

                Item {
                    width: parent.width
                    height: 30
                    anchors.top: seamSpliter.bottom
                    anchors.topMargin: itemHeight
                    Text {
                        id: seamLabel
                        width: lblWidth
                        color: textColor
                        text: qsTr("View")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    Switch{
                        id: seamSwitch
                        anchors.left: seamLabel.right
                        anchors.leftMargin: spacing
                        width: 50
                        height: 30
                        checked: false
                        onClicked: {
                            if(seamSwitch.checked){
								sphericalView.isSeam = true;
                                showSeam();
                            }else{
                                closeSeam();
                            }

                        }
                    }

                    ComboBox {
                        id: cameraList
                        width: parent.width / 4
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.right: parent.right
                        anchors.rightMargin: spacing
                        enabled: false
                        model: ListModel {
                            id: camListModel
                        }
                        onCurrentTextChanged: {
                            if (currentIndex == 0)
                                showSeam()

							if (currentIndex > 0)
							{
								cameraCombo.currentIndex = currentIndex - 1;
								showSeam();
							}
                        }
                    }
                }
            }
        }
    }

    function showSeam()
    {
        var cameraIndex1 = isExposure === true ? cameraCombo.currentIndex + 1: cameraList.currentIndex;

        
        if (qmlMainWindow.enableSeam(cameraIndex1, -20))
        {
            cameraList.enabled = true;
            sphericalView.setSeamIndex(cameraIndex1, -20);
        }
        else
        {
            seamSwitch.checked = false;
        }
    }

    function closeSeam()
    {
        cameraList.enabled = false;
        sphericalView.closeSeam();
    }

    function createTopCameraCombo(){
        cameraModel.clear();

        cameraCnt = qmlMainWindow.getCameraCount();
        for(var i = 0; i < cameraCnt; i++){
            cameraModel.append({"text": "Camera" + (i + 1)})
        }

     }

    function getCameraList()
    {
        camListModel.clear();
        camListModel.append({"text": "All"});
        for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
            camListModel.append({"text": "Camera" + (i + 1)})
        }
    }

    function initSeamSettings()
    {
        sphericalView.isSeam = false;
        if(seamSwitch.checked){
            seamSwitch.checked = false;
            cameraList.currentIndex = 0;
            cameraList.enabled = false;
        }
    }

    function getBlendAndParams()
    {
        blendSettings.getBlendSettings();        
    }

    function getBlendMode()
    {
        var blendMode = qmlMainWindow.getBlendMode();
        if(blendMode)
        {
            featherRadio.checked = false;
            blendRadio.checked = true;
        }
        else
        {
            featherRadio.checked = true;
            blendRadio.checked = false;
        }
		
		isBlend = (blendMode == 1) ? false : true;
    }

	function switchPreview()
	{
		qmlMainWindow.updateCameraView(cameraPreviewObject.camView, cameraCombo.currentIndex)
	}

    function setFeatherMode(){
        levelItem.visible = false;
        isBlend = false;
    }

    function setBlendMode(){
        levelItem.visible = true;
        isBlend = true;
    }

    function getExposureSetting()
    {
        exposureSlider.value = qmlMainWindow.getCameraExposure(cameraCombo.currentIndex);
    }

    function setExposureSetting(){
        qmlMainWindow.setCameraExposure(cameraCombo.currentIndex, exposureText.text);
        qmlMainWindow.reStitch();
    }

    function setCalibSettings()
    {
        // Auto Calibration Parameters
        qmlMainWindow.setFov(hfov.text);
        qmlMainWindow.setLensType(lensTypeCombo.currentIndex);
    }

    function onDetails()
    {
        liveView.createDetailsWindow();
    }

    function onCalculate()
    {
        refreshCameraValues();
        qmlMainWindow.onCalculatorGain();
    }

    function showExposureSetting()
    {
        exposure.visible = true;
        seam.visible = false;
        resetCalibItem.visible = false;        
        //arrowUpItem.visible = false;
        //arrowDownItem.visible = false;
        blendSettingGroup.visible = false;
        blendSettings.visible = false;
        blendSettingSpliter.visible = false;
        arrowUp_blendItem.visible = false;
        arrowDown_blendItem.visible = false;
        nItemCount = 8;
        sphericalView.isSeam = true;
        createTopCameraCombo();
		exposureSlider.value = qmlMainWindow.getCameraExposure(cameraCombo.currentIndex)
		showSeam();
    }

    function showBlendSettings()
    {
        //blend.visible = true;
        exposure.visible = false;
        seam.visible = true;
        resetCalibItem.visible = true;
        
        if(blendSettings.state === "collapsed"){
            arrowUp_blendItem.visible = false;
            arrowDown_blendItem.visible = true;
            nItemCount = 15;
        } else {
            arrowUp_blendItem.visible = true;
            arrowDown_blendItem.visible = false;
            nItemCount = 35;
        }
        

        blendSettings.visible = true;
        blendSettingGroup.visible = true;
        blendSettingSpliter.visible = true;
        applyItem.visible = false;
    }

	function collapseSphericalbox()
    {
        sphericalBox.state = "collapsed";
        blendSettings.state = "collapsed";        
        if (sphericalView.isSeam === false) return;
        initSeamSettings();
		closeSeam();
    }
}
