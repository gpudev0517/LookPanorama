import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2
import "../controls"

ToolWindow{
    id: details
    width: 350
    height: 850
    z: 10
    windowState: qsTr("windowed")
    visible: true

    property string   title: "Rig Template"
    property int      fontSize: 15
    property int      itemHeight: 30
    property int      spacing: 20
    property color    textColor: "#ffffff"
    property var      currentCursorShape: Qt.CrossCursor
    property int      lblWidth: 60

    property int        leftMargin: 20
    property int        rightMargin: 0
    property color      spliterColor: "#555555"
    property int        lblFont: 14
    property bool       isCPoint: false
    property int        cameraCnt
    property var        opacityVal: 0.9
    property int        camWidth: 100
    property int        camHeight: 100
    property int        m_curCameraIndex;
    property bool       m_isReady: false

    Item {
        id: cameraParamsItem
        width : 330
        height: parent.height

        property bool       isHovered : false

        FileDialog {
            id: loadFileDialog
            title: "Load PAC file"
            selectMultiple: false
            nameFilters: [ "PAC file (*.pac)"]
            selectedNameFilter: "All files (*)"
            onAccepted: {
                var pacFilePath = fileUrl.toString().substring(8); // Remove "file:///" prefix

                rigLoadSettingbox.x = (centralItem.width - rigLoadSettingbox.width) / 2 ;
                rigLoadSettingbox.y = 0;
                rigLoadSettingbox.visible = true;
                rigLoadSettingbox.setFilePath(pacFilePath);
                rigLoadSettingbox.initUI();
            }
            onRejected: {
            }
        }

        FileDialog {
            id: saveFileDialog
            title: "Save PAC file"
            selectExisting: false
            selectFolder: false
            selectMultiple: false
            nameFilters: [ "PAC file (*.pac)"]
            selectedNameFilter: "All files (*)"
            onAccepted: {
                var pacFilePath = fileUrl.toString().substring(8); // Remove "file:///" prefix

                rigSaveSettingbox.x = (centralItem.width - rigSaveSettingbox.width) / 2 ;
                rigSaveSettingbox.y = 0;
                rigSaveSettingbox.visible = true;
                rigSaveSettingbox.setFilePath(pacFilePath);
                rigSaveSettingbox.initUI();
            }
            onRejected: {
            }
        }

        FileDialog {
            id: favoriteFileDialog
            title: "Save Favorite file"
            selectExisting: false
            selectFolder: false
            selectMultiple: false
            nameFilters: [ "L3D file (*.l3d)"]
            selectedNameFilter: "All files (*)"
            onAccepted: {
                var favoriteFilePath = fileUrl.toString().substring(8); // Remove "file:///" prefix
                qmlMainWindow.addFavoriteTemplate(favoriteFilePath);
            }
            onRejected: {
            }
        }

        ScrollView {
            id: scrollView
            width: parent.width
            height: parent.height - 80
            verticalScrollBarPolicy: Qt.ScrollBarAlwaysOff
            horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff
            flickableItem.interactive: true


            style: ScrollViewStyle {
                transientScrollBars: false
                handle: Item {
                    implicitWidth: 14
                    implicitHeight: 260
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
                    visible: true
                }

                incrementControl: Rectangle{
                    visible: true
                }
                corner: Rectangle{
                    visible:false
                }

            }

            Item{
                id: groupItem
                width: scrollView.width
                height: scrollView.height + 170

                function convertAngle(angleText)
                {
                    var val = Number(angleText);
                    while (val > 180)
                        val = val - 360;
                    while(val < -180)
                        val = val + 360;
                    return val;
                }

                Item {
                    id: cameraItem
                    width: parent.width
                    height: 30
                    anchors.top: parent.top
                    anchors.topMargin: 10

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
                        width: parent.width / 2
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: cameraText.right
                        anchors.leftMargin: spacing
                        model: ListModel {
                            id: camListModel
                        }

                        onCurrentTextChanged: {
                            m_curCameraIndex = currentIndex;
                            getCameraParams(m_curCameraIndex);
                        }
                    }
                }

                Item {
                    id: lensTypeItem
                    width: parent.width
                    height: 30
                    anchors.top: cameraItem.bottom
                    anchors.topMargin: spacing

                    Text {
                        id: lensTypeText
                        color: textColor
                        width: lblWidth
                        text: qsTr("Lens Type")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: spacing
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: 13
                    }

                    ComboBox {
                        id: lensTypeCombo
                        width: parent.width / 2
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: lensTypeText.right
                        anchors.leftMargin: spacing
                        model: ["Standard", "Fullframe Fisheye", "Circular Fisheye", "CV Standard", "CV Fisheye"]
                        enabled: false
                        onCurrentTextChanged: {
                            qmlMainWindow.setLensType(currentIndex);
                            getCameraParams(m_curCameraIndex);
                        }
                    }
                }

                Item {
                    id: sizeItem
                    width: parent.width
                    height: 30
                    anchors.top: lensTypeItem.bottom
                    anchors.topMargin: spacing

                    Text {
                        id:xText
                        color: textColor
                        width: lblWidth
                        text: qsTr("Width")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont
                    }

                    FlatText {
                        id: widthText
                        width: parent.width * 0.15
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: xText.right
                        anchors.leftMargin: spacing
                        enabled: false
                    }

                    Text {
                        id: yText
                        x: 175
                        color: textColor
                        width: lblWidth
                        text: qsTr("Height")
                        horizontalAlignment: Text.AlignLeft
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont
                    }

                    FlatText {
                        id: heightText
                        width: parent.width * 0.15
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                        enabled: false
                    }
                }

                CameraParameterItem {
                    id: yawItem
                    title: qsTr("Yaw")
                    anchors.top: sizeItem.bottom
                    anchors.topMargin: spacing * 1.5

                    miniumValue: -180
                    maximumValue: 180

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: pitchItem
                    title: qsTr("Pitch")
                    anchors.top: yawItem.bottom

                    miniumValue: -180
                    maximumValue: 180

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: rollItem
                    title: qsTr("Roll")
                    anchors.top: pitchItem.bottom

                    miniumValue: -180
                    maximumValue: 180

                    stepSize: 1

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: fovItem
                    title: qsTr("Fov")
                    anchors.top: rollItem.bottom

                    miniumValue: 0.1
                    maximumValue: 360

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: fovYItem
                    title: qsTr("FocalY")
                    anchors.top: fovItem.bottom

                    miniumValue: 0.1
                    maximumValue: 360

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: k1Item
                    title: qsTr("K1")
                    anchors.top: fovYItem.bottom

                    miniumValue: -2
                    maximumValue: 2

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: k2Item
                    title: qsTr("K2")
                    anchors.top: k1Item.bottom

                    miniumValue: -2
                    maximumValue: 2

                    stepSize: 0.001

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: k3Item
                    title: qsTr("K3")
                    anchors.top: k2Item.bottom

                    miniumValue: -2
                    maximumValue: 2

                    stepSize: 0.0001

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: offsetXItem
                    title: qsTr("OffsetX")
                    anchors.top: k3Item.bottom

                    miniumValue: - camWidth / 2
                    maximumValue: camWidth / 2

                    stepSize: 0.1

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: offsetYItem
                    title: qsTr("OffsetY")
                    anchors.top: offsetXItem.bottom

                    miniumValue: - camWidth / 2
                    maximumValue: camWidth / 2

                    stepSize: 0.1

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: leftItem
                    title: qsTr("Left")
                    anchors.top: offsetYItem.bottom

                    miniumValue: 0
                    maximumValue: qmlMainWindow.xRes

                    stepSize: 1

                    onParameterValueChanged: {
                        if(qmlMainWindow.getLensType() === 0 || qmlMainWindow.getLensType() === 3)
                            rightItem.initializeValue = rightItem.maximumValue - value;
                        setBlendSettings()
                    }
                }

                CameraParameterItem {
                    id: rightItem
                    title: qsTr("Right")
                    anchors.top: leftItem.bottom

                    miniumValue: 0
                    maximumValue: qmlMainWindow.xRes

                    stepSize: 1

                    onParameterValueChanged: {
                        if(qmlMainWindow.getLensType() === 0 || qmlMainWindow.getLensType() === 3)
                            leftItem.initializeValue = leftItem.maximumValue - value;
                        setBlendSettings();
                    }
                }

                CameraParameterItem {
                    id: topItem
                    title: qsTr("Top")
                    anchors.top: rightItem.bottom

                    miniumValue: 0
                    maximumValue: qmlMainWindow.yRes

                    stepSize: 1

                    onParameterValueChanged: {
                        if(qmlMainWindow.getLensType() === 0 || qmlMainWindow.getLensType() === 3)
                            bottomItem.initializeValue = bottomItem.maximumValue - value;
                        setBlendSettings()
                    }
                }

                CameraParameterItem {
                    id: bottomItem
                    title: qsTr("Bottom")
                    anchors.top: topItem.bottom

                    miniumValue: 0
                    maximumValue: qmlMainWindow.yRes

                    stepSize: 1

                    onParameterValueChanged: {
                        if(qmlMainWindow.getLensType() === 0 || qmlMainWindow.getLensType() === 3)
                            topItem.initializeValue = topItem.maximumValue - value;
                        setBlendSettings()
                    }
                }
            }
        }

        Item {
            id: loadAndSaveItem
            height: 30
            width: 330
            anchors.bottom: parent.bottom
            anchors.bottomMargin: spacing

            Item {
                id: resetItem
                anchors.right: loadItem.left
                anchors.rightMargin: 20
                width: 65
                height: 30

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
                    text: "Reset"
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

                        onEntered: resetHoverRect.visible = true
                        onExited: resetHoverRect.visible = false

                        onClicked: {
                            resetTemplate();
                        }
                    }
                }
            }

            Item {
                id: loadItem
                anchors.right: saveItem.left
                anchors.rightMargin: 20
                width: 65
                height: 30

                Rectangle {
                    id: loadHoverRect
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
                    id: loadText
                    z: 1
                    color: "#ffffff"
                    text: "Load"
                    font.pointSize: 11
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent
                }

                Rectangle {
                    id: loadRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id: loadMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true

                        onEntered: loadHoverRect.visible = true
                        onExited: loadHoverRect.visible = false

                        onClicked: {
                            loadFileDialog.open();
                        }
                    }
                }
            }

            Item {
                id: saveItem
                width: 65
                height: 30
                anchors.right: favoriteItem.left
                anchors.rightMargin: 20

                Rectangle {
                    id: saveHoverRect
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
                    id: saveText
                    z: 1
                    color: "#ffffff"
                    text: "Save"
                    font.pointSize: 11
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent
                }

                Rectangle {
                    id: saveRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id: saveMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true

                        onEntered: saveHoverRect.visible = true
                        onExited: saveHoverRect.visible = false

                        onClicked: {
                            saveFileDialog.open();
                        }
                    }
                }
            }

            Item {
                id: favoriteItem
                anchors.right: parent.right
                anchors.rightMargin: rightMargin
                width: 65
                height: 30

                Rectangle {
                    id: favoriteHoverRect
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
                    id: favoriteText
                    z: 1
                    color: "#ffffff"
                    text: "Favorite"
                    font.pointSize: 11
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent
                }

                Rectangle {
                    id: favoriteRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id: favoriteMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true

                        onEntered: favoriteHoverRect.visible = true
                        onExited: favoriteHoverRect.visible = false

                        onClicked: {
                            favoriteFileDialog.open();
                        }
                    }
                }
            }

        }

        function appendCameraCombo(){
            camListModel.clear();

            for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
                camListModel.append({"text": "Camera" + (i + 1)})
            }

            cameraCombo.currentIndex = 0;
        }

        function getCameraParams(curCameraIndex){

            if(!qmlMainWindow.start) return;

            m_isReady = false;

            // check lensType
            var lensType = qmlMainWindow.getLensType(curCameraIndex);
            camWidth = qmlMainWindow.getWidth(curCameraIndex);
            camHeight = qmlMainWindow.getHeight(curCameraIndex);

            yawItem.setInitialValue(qmlMainWindow.getYaw(curCameraIndex))
            pitchItem.setInitialValue(qmlMainWindow.getPitch(curCameraIndex))
            rollItem.setInitialValue(qmlMainWindow.getRoll(curCameraIndex))
            fovItem.setInitialValue(qmlMainWindow.getFov(curCameraIndex))
            fovYItem.setInitialValue(qmlMainWindow.getFovy(curCameraIndex))
            k1Item.setInitialValue(qmlMainWindow.getK1(curCameraIndex))
            k2Item.setInitialValue(qmlMainWindow.getK2(curCameraIndex))
            k3Item.setInitialValue(qmlMainWindow.getK3(curCameraIndex))
            offsetXItem.setInitialValue(qmlMainWindow.getOffsetX(curCameraIndex))
            offsetYItem.setInitialValue(qmlMainWindow.getOffsetY(curCameraIndex))
            leftItem.setInitialValue(qmlMainWindow.getLeft(curCameraIndex))
            rightItem.setInitialValue(qmlMainWindow.getRight(curCameraIndex))
            topItem.setInitialValue(qmlMainWindow.getTop(curCameraIndex))
            bottomItem.setInitialValue(qmlMainWindow.getBottom(curCameraIndex))

            // check lensType
            //var lensType = qmlMainWindow.getLensType();
            lensTypeCombo.currentIndex = lensType;
            if (lensType == 3) {
                // LensType_opencvLens_Standard
                fovItem.title = "FoV";
                fovYItem.visible = false;
                k1Item.anchors.top = fovItem.bottom;
            } else if (lensType == 4) {
                // LensType_opencvLens_Fisheye
                fovItem.title = "FoVX";
                fovYItem.title = "FoVY";
                fovYItem.visible = true;
                k1Item.anchors.top = fovYItem.bottom;
            } else {
                fovItem.title = "FoV";
                fovYItem.title = false;
                fovYItem.visible = false;
                k1Item.anchors.top = fovItem.bottom;
            }

            // Get blend setting values
            widthText.text = qmlMainWindow.getWidth(curCameraIndex);
            heightText.text = qmlMainWindow.getHeight(curCameraIndex);

            m_isReady = true;
        }

        function setCameraParams(){

            if(!qmlMainWindow.start || !m_isReady) return;

            qmlMainWindow.setWidth(widthText.text, m_curCameraIndex);
            qmlMainWindow.setHeight(heightText.text, m_curCameraIndex);
            qmlMainWindow.setYaw(yawItem.value, m_curCameraIndex);
            qmlMainWindow.setPitch(pitchItem.value, m_curCameraIndex);
            qmlMainWindow.setRoll(rollItem.value, m_curCameraIndex);
            qmlMainWindow.setFov(fovItem.value, m_curCameraIndex);
            qmlMainWindow.setFovy(fovYItem.value, m_curCameraIndex);
            qmlMainWindow.setK1(k1Item.value, m_curCameraIndex);
            qmlMainWindow.setK2(k2Item.value, m_curCameraIndex);
            qmlMainWindow.setK3(k3Item.value, m_curCameraIndex);
            qmlMainWindow.setOffsetX(offsetXItem.value, m_curCameraIndex);
            qmlMainWindow.setOffsetY(offsetYItem.value, m_curCameraIndex);

            qmlMainWindow.reStitch(true);
        }

        function insertCamName(){
            cameraModel.clear();
            for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
                cameraModel.append({"text": "Camera" + (i + 1)})
            }
        }

   }

    function resetTemplate() {
        qmlMainWindow.onCancelCameraSettings();
        qmlMainWindow.reStitch(true);
        for(var i = 0; i < qmlMainWindow.getCameraCount() ; i++){
            getCameraParams(i);
        }
        getCameraParams(0);
        cameraCombo.currentIndex = 0;
    }

    function getCameraParams(curCameraIndex){
        m_curCameraIndex = curCameraIndex;
        cameraParamsItem.getCameraParams(curCameraIndex);
    }

    function setCameraParams(){
        cameraParamsItem.setCameraParams();
    }

    function initCameraParms(){
        cameraParamsItem.appendCameraCombo();
        getCameraParams(cameraCombo.currentIndex);
    }

    function setBlendSettings(){

        if(!qmlMainWindow.start || !m_isReady) return;

        qmlMainWindow.setLeft(leftItem.value, m_curCameraIndex);
        qmlMainWindow.setRight(rightItem.value, m_curCameraIndex);
        qmlMainWindow.setTop(topItem.value, m_curCameraIndex)
        qmlMainWindow.setBottom(bottomItem.value, m_curCameraIndex);

        qmlMainWindow.reStitch(true);
    }
}
