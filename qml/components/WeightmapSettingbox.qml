import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.0

ToolWindow{
    id: details
    width: 330
    height: 500
    z: 10
    windowState: qsTr("windowed")
    visible: false

    property string   title: "Weightmap Settings"
    property int      fontSize: 15
    property int      itemHeight: 30
    property int      spacing: 20
    property color    textColor: "#ffffff"
    property var      currentCursorShape: Qt.CrossCursor
    property int      lblWidth: 60
    property bool     isEditweightmap: true
	property CameraParamsSettingbox cameraParamsSettingbox;
    property int      m_weightMapMode: 1

    property bool     isMirror: false
    property int      cameraComboWidth:200

    property var      rightCameraIndexes: []
    property var      leftCameraIndexes: []

    property var eyeMode: {
          "DEFAULT": 0,
          "LEFT": 1,
          "RIGHT": 2,
          "BOTH": 3,
          "MIRROR": 4
        };

    Item {
        id: brushItem
        width: parent.width
        height: itemHeight
        anchors.top: parent.top
        anchors.topMargin: spacing

        property bool isIncrement: true
        Text {
            id: brushText
            color: textColor
            width: lblWidth
            text: qsTr("Brush")
            horizontalAlignment: Text.AlignLeft
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }

        Image {
            id:  brushImage
            y: 5
            anchors.left: brushText.right
            anchors.leftMargin: spacing
            width: 20
            height: 20
            visible: true
            source: "../../resources/ico_brush.png"
            fillMode: Image.PreserveAspectFit

            MouseArea {
                id: brushMouse
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                   // brushSelectRectangle.visible = true;
                    cameraItem.enabled = true;
                    radiusItem.enabled = true;
                    strengthItem.enabled = true;
                    falloffItem.enabled = true;
                    isEditweightmap = true;//draw Weightmap
                    placementImage.source = "../../resources/ico_placement.png"
                    currentCursorShape = Qt.CrossCursor;
                }
            }
        }

        Item {
            id: alphaPlus
            anchors.left: brushImage.right
            anchors.leftMargin: spacing
            y: 5
            property int screenNum: 1
            property bool isSelected: false
            width: 28
            height: 20
            Rectangle {
                id: alphaPlusRectangle
                color: "#171717"
                border.color: brushItem.isIncrement ? "#4e8d15": "#00000000"
                implicitWidth: parent.width
                implicitHeight: parent.height

                Text {
                    anchors.centerIn: parent
                    text: "+"
                    color: "white"
                    font.pixelSize: 15
                }
            }

            MouseArea{
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                    alphaPlusRectangle.border.color = "#4e8d15";
                    alphaMinRectangle.border.color = "#00000000";
                    brushItem.isIncrement = true;
                    setDrawWeightmapSettings();
                }
            }
        }

        Item {
            id: alphaMin
            anchors.left: alphaPlus.right
            anchors.leftMargin: spacing / 2
            y: 5
            property int screenNum: 1
            property bool isSelected: false
            width: 28
            height: 20
            Rectangle {
                id: alphaMinRectangle
                color: "#171717"
                border.color: !brushItem.isIncrement ? "#4e8d15": "#00000000"
                implicitWidth: parent.width
                implicitHeight: parent.height

                Text {
                    anchors.centerIn: parent
                    text: "-"
                    color: "white"
                    font.pixelSize: 15
                }
            }

            MouseArea{
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                    alphaMinRectangle.border.color = "#4e8d15";
                    alphaPlusRectangle.border.color = "#00000000";
                    brushItem.isIncrement = false;
                   setDrawWeightmapSettings();
                }
            }
        }

     }

    Item {
        id: placementItem
        width: parent.width
        height: itemHeight
        anchors.top: brushItem.bottom
        anchors.topMargin: spacing / 2
        Text {
            id: placementText
            color: textColor
            width: lblWidth
            text: qsTr("Placement")
            horizontalAlignment: Text.AlignLeft
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }

        Image {
            id:  placementImage
            anchors.left: placementText.right
            anchors.leftMargin: spacing
            width: 30
            height: 30
            visible: true
            source: "../../resources/ico_placement.png"
            fillMode: Image.PreserveAspectFit

            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                    cameraItem.enabled = false;
                    radiusItem.enabled = false;
                    strengthItem.enabled = false;
                    falloffItem.enabled = false;
                    isEditweightmap = false;//placement
                    placementImage.source = "../../resources/ico_select_placement.png";
                   currentCursorShape = Qt.PointingHandCursor;
                }
            }
        }
     }

	Item {
        id: viewModeItem
        width: parent.width
        height: itemHeight
        anchors.top: placementItem.bottom
        anchors.topMargin: spacing
        
        ExclusiveGroup {
            id: viewModeGroup
        }

        RadioButton {
            id: paintRadioButton
            checked: true
            exclusiveGroup: viewModeGroup
            anchors.left: parent.left
            anchors.leftMargin: spacing
            onCheckedChanged: {
                if(checked) {
                    viewModeItem.setWeightMapMode(1)
                }
            }
        }

        Text {
            id: paintLabel
            anchors.left: paintRadioButton.right
            anchors.leftMargin: spacing / 5
            anchors.verticalCenter: parent.verticalCenter
            color: "#ffffff"
            text: qsTr("Weight")
            font.pixelSize: 13
        }


        RadioButton {
            id: viewRadioButton
            exclusiveGroup: viewModeGroup
            anchors.left: paintLabel.right
            anchors.leftMargin: spacing
            onCheckedChanged: {
                if(checked) {
                    viewModeItem.setWeightMapMode(2)
                }
            }
        }

        Text {
            id: viewLabel
            anchors.left: viewRadioButton.right
            anchors.leftMargin: spacing / 5
            anchors.verticalCenter: parent.verticalCenter
            color: "#ffffff"
            text: qsTr("Color")
            font.pixelSize: 13
        }

        RadioButton {
            id: overlapRadioButton
            exclusiveGroup: viewModeGroup
            anchors.left: viewLabel.right
            anchors.leftMargin: spacing
            onCheckedChanged: {
                if(checked) {
                    viewModeItem.setWeightMapMode(3)
                }
            }
        }

        Text {
            id: overlapLabel
            anchors.left: overlapRadioButton.right
            anchors.leftMargin: spacing / 5
            anchors.verticalCenter: parent.verticalCenter
            color: "#ffffff"
            text: qsTr("Overlap")
            font.pixelSize: 13
        }

        function setWeightMapMode(weightMode){			
            m_weightMapMode = weightMode;
            qmlMainWindow.setWeightMapPaintingMode(weightMode);
        }
     }

    Item {
        id: eyeItem
        width: parent.width
        height: itemHeight
        anchors.top: viewModeItem.bottom
        anchors.topMargin: spacing

        ExclusiveGroup {
            id: eyeGroup
        }

        RadioButton {
            id: leftButton
            checked: true
            exclusiveGroup: eyeGroup
            anchors.left: parent.left
            anchors.leftMargin: spacing
            onCheckedChanged: {
                if (checked) {
                    appendLeftCameraCombo()
                    setWeightmapWith(leftCameraIndexes[0], -20)
                    showSeamWith(leftCameraIndexes[0])

                    cameraCombo.currentIndex = 0
                    cameraCombo_.currentIndex = 0
                }
            }
        }

        Text {
            id: leftLabel
            anchors.left: leftButton.right
            anchors.leftMargin: spacing / 5
            anchors.verticalCenter: parent.verticalCenter
            color: "#ffffff"
            text: qsTr("Left")
            font.pixelSize: 13
        }


        RadioButton {
            id: rightButton
            exclusiveGroup: eyeGroup
            anchors.left: leftLabel.right
            anchors.leftMargin: spacing / 2
            onCheckedChanged: {
                if (checked) {
                    appendRightCameraCombo()
                    setWeightmapWith(rightCameraIndexes[0], -20)
                    showSeamWith(rightCameraIndexes[0])

                    cameraCombo.currentIndex = 0
                    cameraCombo_.currentIndex = 0
                }
            }
        }

        Text {
            id: rightLabel
            anchors.left: rightButton.right
            anchors.leftMargin: spacing / 5
            anchors.verticalCenter: parent.verticalCenter
            color: "#ffffff"
            text: qsTr("Right")
            font.pixelSize: 13
        }

        RadioButton {
            id: bothButton
            exclusiveGroup: eyeGroup
            anchors.left: rightLabel.right
            anchors.leftMargin: spacing / 2
            onCheckedChanged: {
                if (checked) {
                    appendCameraCombo()
                    cameraCombo.currentIndex = 0
                    cameraCombo_.currentIndex = 0
                    setDrawWeightmapSettings();
                    showSeam()
                }
            }
        }

        Text {
            id: bothLabel
            anchors.left: bothButton.right
            anchors.leftMargin: spacing / 5
            anchors.verticalCenter: parent.verticalCenter
            color: "#ffffff"
            text: qsTr("Both")
            font.pixelSize: 13
        }

        RadioButton {
            id: mirrorRadio
            exclusiveGroup: eyeGroup
            anchors.left: bothLabel.right
            anchors.leftMargin: spacing / 2

            onCheckedChanged: {
                isMirror = !isMirror

                if (checked) {

                    camListModel.clear()
                    for (var i = 0; i < leftCameraIndexes.length; i++) {
                        camListModel.append({"text": "Camera" + (leftCameraIndexes[i] +1)})
                    }

                    camListModel_.clear()
                    for (var i = 0; i < rightCameraIndexes.length; i++) {
                        camListModel_.append({"text": "Camera" + (rightCameraIndexes[i] +1)})
                    }

                    setWeightmapWith(leftCameraIndexes[0], rightCameraIndexes[0])
                    getCameraParams(0)
                    showSeamWith(leftCameraIndexes[0], rightCameraIndexes[0])
                }
            }
        }

        Text {
            id: mirrorText
            anchors.left: mirrorRadio.right
            anchors.leftMargin: spacing / 5
            anchors.verticalCenter: parent.verticalCenter
            color:"#ffffff"
            text: qsTr("Mirror")
            font.pixelSize: 13
        }
     }

    Item {
        id: cameraItem
        width: parent.width
        height: itemHeight
        anchors.top: eyeItem.bottom
        anchors.topMargin: spacing
        Text {
            id: cameraText
            color: textColor
            width: lblWidth
            text: qsTr("Camera")
            horizontalAlignment: Text.AlignLeft
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13

        }

        ComboBox {
            id: cameraCombo
            width: isMirror?100: 200//cameraComboWidth
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: cameraText.right
            anchors.leftMargin: spacing /2
            model: ListModel {
                id: camListModel
            }            

            onCurrentTextChanged: {

                if (leftButton.checked) {
                    setWeightmapWith(leftCameraIndexes[currentIndex], -20)
                    getCameraParams(currentIndex)
                    showSeamWith(leftCameraIndexes[currentIndex])
                }

                if (rightButton.checked) {
                    setWeightmapWith(rightCameraIndexes[currentIndex], -20)
                    getCameraParams(currentIndex)
                    showSeamWith(rightCameraIndexes[currentIndex])
                }

                if (bothButton.checked) {
                    setDrawWeightmapSettings()
                    showSeam()
                    getCameraParams(currentIndex)
                }

                if (mirrorRadio.checked) {
                    setWeightmapWith(leftCameraIndexes[currentIndex], rightCameraIndexes[cameraCombo_.currentIndex])
                    getCameraParams(currentIndex)
                    showSeamWith(leftCameraIndexes[currentIndex], rightCameraIndexes[cameraCombo_.currentIndex])
                }
            }
        }

        ComboBox {
            id: cameraCombo_
            width: 100
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: cameraCombo.right
            anchors.leftMargin: spacing /2
            visible: isMirror

            model: ListModel {
                id: camListModel_
            }

            onCurrentTextChanged: {
                if (mirrorRadio.checked) {
                    setWeightmapWith(leftCameraIndexes[cameraCombo.currentIndex], rightCameraIndexes[currentIndex])
                    getCameraParams(currentIndex)
                    showSeamWith(leftCameraIndexes[cameraCombo.currentIndex], rightCameraIndexes[currentIndex])
                }
            }
        }
     }

    Item {
        id: radiusItem
        width: parent.width
        height: itemHeight
        anchors.top: cameraItem.bottom
        anchors.topMargin: spacing
        Text {
            id: radiusText
            color: textColor
            width: lblWidth
            text: qsTr("Radius")
            horizontalAlignment: Text.AlignLeft
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }

        Slider {
            id: radiusSlider
            width: parent.width / 2.5
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: radiusText.right
            anchors.leftMargin: spacing
            stepSize: 1
            minimumValue: 5
            maximumValue: 100
            onPressedChanged: {
                if(!pressed){
                    setDrawWeightmapSettings();
                }

            }
        }

        Text {
            id: radius
            color: textColor
            text: radiusSlider.value
            horizontalAlignment: Text.AlignLeft
            anchors.right:  parent.right
            anchors.rightMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }

     }

    Item {
        id: strengthItem
        width: parent.width
        height: itemHeight
        anchors.top: radiusItem.bottom
        anchors.topMargin: spacing
        Text {
            id: strengthText
            color: textColor
            width: lblWidth
            text: qsTr("Strength")
            horizontalAlignment: Text.AlignLeft
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13

        }

        Slider {
            id: strengthSlider
            width: parent.width / 2.5
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: strengthText.right
            anchors.leftMargin: spacing
            stepSize: 1
            minimumValue: 0
            maximumValue: 100
            onPressedChanged: {
                if(!pressed){
                    setDrawWeightmapSettings();
                }
            }
        }

        Text {
            id: strength
            color: textColor
            text: strengthSlider.value
            horizontalAlignment: Text.AlignLeft
            anchors.right: parent.right
            anchors.rightMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }
     }

    Item {
        id: falloffItem
        width: parent.width
        height: itemHeight
        anchors.top: strengthItem.bottom
        anchors.topMargin: spacing
        Text {
            id: falloffText
            color: textColor
            width: lblWidth
            text: qsTr("Fall off")
            horizontalAlignment: Text.AlignLeft
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13

        }

        Slider {
            id: fallOffSlider
            width: parent.width / 2.5
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: falloffText.right
            anchors.leftMargin: spacing
            stepSize: 1
            minimumValue: 0
            maximumValue: 100
            onPressedChanged: {
                if(!pressed){
                  setDrawWeightmapSettings();
                }
            }
        }

        Text {
            id: falloff
            color: textColor
            text: fallOffSlider.value
            horizontalAlignment: Text.AlignLeft
            anchors.right: parent.right
            anchors.rightMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }

     }

    Item{
        id: undoItem
        anchors.top: falloffItem.bottom
        anchors.topMargin: spacing
        anchors.right: redoItem.left
        width: 68
        height: 48
        Rectangle{
            id:  undoHoverRectangle
            width: parent.width
            height: parent.height
            color: "#353535"
            visible: false
        }
        MouseArea {
            id: undoMouseArea
            x: undoHoverRectangle.x
            z: 2
            width: undoHoverRectangle.width
            height: undoHoverRectangle.height
            hoverEnabled: true
            onEntered: undoHoverRectangle.visible = true;
            onExited: undoHoverRectangle.visible = false;
            onClicked: {
                //undo function
				qmlMainWindow.weightmapUndo();
            }
        }
        Image {
            id: undoImage
            z: 1
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            width: 35
            height: 35
            fillMode: Image.PreserveAspectFit
            source: "../../resources/ico_undo.png"
            visible: false
        }

        Image {
            id: greyUndoImage
            z: 1
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            width: 35
            height: 35
            fillMode: Image.PreserveAspectFit
            source: "../../resources/ico_grey_undo.png"
        }
    }

    Item{
        id: redoItem
        anchors.top: falloffItem.bottom
        anchors.topMargin: spacing
        anchors.right: parent.right
        anchors.rightMargin: (parent.width - width * 2) / 2
        width: 68
        height: 48
        Rectangle{
            id:  redoHoverRectangle
            width: parent.width
            height: parent.height
            color: "#353535"
            visible: false
        }
        MouseArea {
            id: redoMouseArea
            x: redoHoverRectangle.x
            z: 2
            width: redoHoverRectangle.width
            height: redoHoverRectangle.height
            hoverEnabled: true
            onEntered: redoHoverRectangle.visible = true;
            onExited: redoHoverRectangle.visible = false;
            onClicked: {
                //redo function
				qmlMainWindow.weightmapRedo();
            }
        }
        Image {
            id: redoImage
            z: 1
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            width: 35
            height: 35
            fillMode: Image.PreserveAspectFit
            source: "../../resources/ico_redo.png"
            visible: false
        }

        Image {
            id: greyRedoImage
            z: 1
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            width: 35
            height: 35
            fillMode: Image.PreserveAspectFit
            source: "../../resources/ico_grey_redo.png"
        }
    }

    function appendCameraCombo(){

        camListModel.clear()

        for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
            camListModel.append({"text": "Camera" + (i + 1)})
        }

        leftCameraIndexes = []
        for (var i = 0; i < qmlMainWindow.getCameraCount(); i++) {
            if (qmlMainWindow.isLeftEye(i)) {
                leftCameraIndexes.push(i)
            }
        }

        rightCameraIndexes = []
        for (var i = 0; i < qmlMainWindow.getCameraCount(); i++) {
            if (qmlMainWindow.isRightEye(i)) {
                rightCameraIndexes.push(i)
            }
        }
     }

    function appendLeftCameraCombo() {
        camListModel.clear()
        leftCameraIndexes = []

        for (var i = 0; i < qmlMainWindow.getCameraCount(); i++) {
            if (qmlMainWindow.isLeftEye(i)) {
                camListModel.append({"text": "Camera" + (i +1)})
                leftCameraIndexes.push(i)
            }
        }
    }

    function appendRightCameraCombo() {
        camListModel_.clear()
        rightCameraIndexes = []
        camListModel.clear()

        for (var i = 0; i < qmlMainWindow.getCameraCount(); i++) {
            if (qmlMainWindow.isRightEye(i)) {
                if (rightButton.checked)
                    camListModel.append({"text": "Camera" + (i + 1)})
                else if (mirrorRadio.checked)
                    camListModel_.append({"text": "Camera" + (i + 1)})
                rightCameraIndexes.push(i)
            }
        }
    }


	function reverseViewModeSwitchStatus() {		
        qmlMainWindow.setWeightMapPaintingMode(m_weightMapMode);
	}

    function initWeightmapSettings(){        

        alphaPlusRectangle.border.color = "#4e8d15";
        alphaMinRectangle.border.color = "#00000000";
        brushItem.isIncrement = true;
        radiusSlider.value = 20;
        strengthSlider.value = (strengthSlider.maximumValue + strengthSlider.minimumValue) / 2 ;
        fallOffSlider.value = fallOffSlider.maximumValue / 2;        
        paintRadioButton.checked = true;
        m_weightMapMode = 1;
        viewModeItem.setWeightMapMode(m_weightMapMode);
		cameraCombo.currentIndex = 0;
		sphericalView.isSeam = true;
		showSeam();

        if (qmlMainWindow.isStereo()){
            eyeItem.visible = true;
            bothButton.checked = true;
        } else {
            eyeItem.visible = false;
        }
    }

    function setDrawWeightmapSettings(){
        var eyeIndex = eyeMode.DEFAULT;
        if (qmlMainWindow.isStereo()){
            if (leftButton.checked)
                eyeIndex = eyeMode.LEFT;
            else if (rightButton.checked)
                eyeIndex = eyeMode.RIGHT;
            else if (bothButton.checked) {
                eyeIndex = eyeMode.BOTH;
            }
            else if (mirrorRadio.checked)
                eyeIndex = eyeMode.MIRROR;

            eyeItem.enabled = true;

        } else {
            eyeItem.enabled = false;            
        }
       qmlMainWindow.setDrawWeightMapSetting(cameraCombo.currentIndex,cameraCombo_.currentIndex, radiusSlider.value,strengthSlider.value,fallOffSlider.value,brushItem.isIncrement, eyeIndex);
    }

    function setWeightmapWith(numCameraIndex, numCameraIndex_){
        var eyeIndex = eyeMode.MIRROR;
        if (qmlMainWindow.isStereo()){
            if (leftButton.checked)
                eyeIndex = eyeMode.LEFT;
            else if (rightButton.checked)
                eyeIndex = eyeMode.RIGHT;
            else if (bothButton.checked)
                eyeIndex = eyeMode.BOTH;
            else if (mirrorRadio.checked)
                eyeIndex = eyeMode.MIRROR;
            eyeItem.enabled = true;

        } else {
            eyeIndex = eyeMode.LEFT
            eyeItem.enabled = false;
        }


        qmlMainWindow.setDrawWeightMapSetting(numCameraIndex,numCameraIndex_, radiusSlider.value,strengthSlider.value,fallOffSlider.value,brushItem.isIncrement, eyeIndex);
    }

	function getCameraParams(curIndex) {
		if (curIndex < 0)
			curIndex = cameraCombo.currentIndex;
		cameraParamsSettingbox.getCameraParams(curIndex);
	}

    function setUndoStatus(weightMapEditUndoStatus){

        if (weightMapEditUndoStatus){
            undoImage.visible = true;
            greyUndoImage.visible = false;
            undoMouseArea.enabled = true;

        } else {
            undoImage.visible = false;
            greyUndoImage.visible = true;
			undoHoverRectangle.visible = false;
            undoMouseArea.enabled = false;
        }

    }

    function setRedoStatusRedo(weightMapEditRedoStatus) {
        if (weightMapEditRedoStatus) {
            redoImage.visible = true;
            greyRedoImage.visible = false;
            redoMouseArea.enabled = true;
        } else {
            redoImage.visible = false;
            greyRedoImage.visible = true;
			redoHoverRectangle.visible = false;
            redoMouseArea.enabled = false;
        }
    }

    function updateCameraIndex(index) {
        cameraCombo.currentIndex = index;
    }

    function closeWeightmapSettingbox() {
        details.visible = false;
        sphericalView.hideSeamLabels()
    }

    function showSeam() {
        var cameraIndex1 = cameraCombo.currentIndex + 1;
        var cameraIndex2 = cameraCombo_.currentIndex + 1;


        if (qmlMainWindow.enableSeam(cameraIndex1, cameraIndex2))
        {
            sphericalView.setSeamIndex(cameraIndex1, cameraIndex2);
        }
    }

    function showSeamWith(numCameraIndex1, numCameraIndex2) {
        var cameraIndex1 = numCameraIndex1 + 1
        var cameraIndex2 = numCameraIndex2 + 1

        if (qmlMainWindow.enableSeam(cameraIndex1, cameraIndex2))
        {
            sphericalView.setSeamIndex(cameraIndex1, cameraIndex2);
        }
    }
}
