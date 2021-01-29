import QtQuick 2.5
import QtQuick.Controls 1.4 //2.0
import QtQuick.Controls.Styles 1.4
import QtQuick.Controls.Styles.Flat 1.0 as Flat
//import QtQuick.layouts 1.1
import "../"
import "."
import QtQuick.Layouts 1.1

import QtQuick.Window 2.2
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import "../controls"

ToolWindow{
    id: singleCalib
    width: 640
    height: 540
    z: 10
    windowState: qsTr("windowed")
    visible: isVisibled


    property string   title: "Single Camera Calibration Toolbox"
    property int      fontSize: 20
    property int      itemHeight: 30
    property int      spacing: 20
    property color    textColor: "#ffffff"
    property var      currentCursorShape: Qt.CrossCursor
    property int      lblWidth: 60
    property bool     isEditweightmap: true
    property bool     isVisibled: true

    property var      wModel: [4, 5, 6, 7, 8, 9, 10, 11, 12]
    property var      hModel: [3, 4, 5, 6, 7, 8, 9, 10]
    property var      snapShotsModel: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    property ListModel camListModel: ListModel{}
    property int      txtFontSize: 14
    property string   txtFontType: "Verdana"
    property int      btnSize: 200
    property int      spaceSize: 10

    property int curCamIndex: comboBoxCamera.currentIndex

    onClosing: {
        singleCalib.visible = false
        qmlMainWindow.stopSingleCapture()
        qmlMainWindow.finishSingleCameraCalibration()
    }

    Rectangle
    {
        anchors.rightMargin: 0
        anchors.bottomMargin: 0
        anchors.leftMargin: 0
        anchors.topMargin: 0
        anchors.fill: parent
        color: "#171717"
        opacity: 0.9
        Rectangle {
            id: rowLayout1
            width: parent.width
            height: 40
            transformOrigin: Item.TopRight
            Layout.fillHeight: false
            Layout.fillWidth: false

            color: "#171717"
            opacity: 0.9

            Text {
                id: textCamera
                x: 20
                y: 3
                width: 50
                height: parent.height
                text: qsTr("Camera")
                color: textColor
                //verticalAlignment: Text.AlignVCenter
                //horizontalAlignment: Text.AlignHCenter
                Layout.fillWidth: false
                font.pixelSize:txtFontSize
                font.family: txtFontType
            }

            ComboBox {
                id: comboBoxCamera
                x: textCamera.x + textCamera.width + 30
                width: 150
                //height: 100
                Layout.fillHeight: false
                Layout.fillWidth: false
                model:camListModel
                currentIndex: 0

                onCurrentIndexChanged: {
                    qmlMainWindow.startWithLiveGrabber(comboBoxCamera.currentIndex);
                    //qmlMainWindow.updateCameraView(calibCamView.camView, currentIndex)
                }
            }

            Text {
                id: textLensType
                text: qsTr("Lens Type")
                color: textColor
                x:comboBoxCamera.x + comboBoxCamera.width + 90
                y: 3
                width: 50
                height: parent.height
                horizontalAlignment: Text.AlignRight
                Layout.fillWidth: false
                font.pixelSize: txtFontSize
                font.family: txtFontType
            }


            ComboBox {
                id: comboBoxLensType
                x: textLensType.x + textLensType.width + 30
                width: btnSize
                //height: 100
                Layout.fillWidth: false
                Layout.fillHeight: false
                currentIndex: 1
                model: ["OpenCV Standard Lens", "OpenCV Fisheye Lens"]
            }

        }

        Rectangle {
            id: rowLayout2
            x:0
            y:rowLayout1.height
            width: parent.width
            height: 40
            Layout.fillHeight: false
            Layout.fillWidth: false
            color: "#171717"
            opacity: 0.9

            Text {
                id: textBoardSize
                x: 15
                y: 1
                text: qsTr("Board Size")
                color: textColor
                font.pixelSize: txtFontSize
                font.family: "Verdana"
                verticalAlignment: Text.AlignBottom
                horizontalAlignment: Text.AlignLeft
                textFormat: Text.AutoText
                wrapMode: Text.NoWrap
            }

            Text {
                id: textW
                x: textBoardSize.x + textBoardSize.width + 30
                y:3
                text: "W"
                color: textColor
                width: 20

                Layout.fillWidth: false
                font.pixelSize: txtFontSize
                font.family: txtFontType
            }

            ComboBox {
                id: comboW
                x: textW.x + textW.width + 5
                width: 60
                enabled: true
                Layout.rowSpan: 1
                Layout.fillWidth: false

                currentIndex: 4
                model: wModel
            }

            Text {
                id: textH
                x: comboW.x + comboW.width + 20
                y:3
                width: 20
                text: qsTr("H")
                color: textColor

                Layout.fillWidth: false
                font.pixelSize: txtFontSize
                font.family: txtFontType
            }

            ComboBox {
                id: comboH
                x: textH.x + textH.width + 5
                width: 60

                Layout.fillWidth: true
                currentIndex: 3
                model: hModel
            }

            Text {
                id: textSnapShotNum
                x: comboH.x + comboH.width + 70
                y: 3
                text: qsTr("Auto-Snapshots")
                color: textColor
                width: 120
                Layout.fillWidth: false
                font.pixelSize: txtFontSize
                font.family: txtFontType
            }

            ComboBox {
                id: comboSnapShots
                x: textSnapShotNum.x + textSnapShotNum.width + 10
                width: 85

                Layout.fillWidth: true

                currentIndex: 6
                model: snapShotsModel
            }
        }

        Rectangle {
            id: videoScreen
            y: rowLayout2.y + rowLayout2.height
            width: parent.width
            height: 320//360
            color: "#171717"
            opacity: 0.9
            Layout.fillHeight: false
            //Layout.preferredWidth: -1
            Layout.fillWidth: false
            property string progressColor: "#373737"


            Rectangle {
                x: 0
                y: 0
                width: parent.width - 20
                height: parent.height

                MCCalibrationCameraView{
                    id: calibCamView
                    anchors.fill: parent
                }
            }

            ProgressBar {
                id: progress
                x: parent.width - 20
                y: 0
                width: 10
                height :parent.height
                value: 0
                maximumValue:  1000
                minimumValue:  0

                style: ProgressBarStyle {
                    background: Rectangle {
                        radius: 2
                        color: "#171717"
                        opacity: 0.9
                        border.color: "#171717"
                        border.width: 1
                        implicitWidth: 200
                        implicitHeight: 10
                    }
                    progress: Rectangle {
                        id: vProgress
                        color: videoScreen.progressColor
                        border.color: "steelblue"
                    }
                }

                orientation: Qt.Vertical
            }
        }

        Rectangle {
            id: rowLayout3
            y: videoScreen.y + videoScreen.height
            width: parent.width
            height: 40

            color: "#171717"
            opacity: 0.9
            Label {
                id: messageText
                anchors.fill: parent
                font.pixelSize: 17
                color: "lightgray"
                text: ""
                font.family: "Verdana"
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }


        Rectangle {
            id: rowLayout4
            x:0
            y: rowLayout3.y + rowLayout3.height
            width: parent.width
            height: 60
            Layout.fillHeight: false
            Layout.fillWidth: false

            color: "#171717"
            opacity: 0.9

            property bool isStartCapture: true

            Item {
                id: captureItem
                Layout.fillWidth: true

                width: btnSize
                height: 50
                x:  80
                y: 5

                Rectangle {
                    id: captureHoverRect
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
                    id: captureText
                    color: "#ffffff"
                    text: rowLayout4.isStartCapture? "Start Capture": "Stop Capture"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent



                }

                Rectangle {
                    id: captureRect
                    width: parent.width
                    height: parent.height
                    color: "#373737"

                    MouseArea {
                        id: captureMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                captureHoverRect.visible = true;
                            }else{
                                captureHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            if (rowLayout4.isStartCapture) {
                                rowLayout4.isStartCapture = !rowLayout4.isStartCapture

                                 calibrateMouseArea.visible = false
                                 comboBoxCamera.enabled = false
                                 comboBoxLensType.enabled = false
                                 comboW.enabled = false
                                 comboH.enabled = false
                                 comboSnapShots.enabled = false

                                 qmlMainWindow.setSingleParams(comboBoxCamera.currentIndex,comboBoxLensType.currentIndex, wModel[comboW.currentIndex],hModel[comboH.currentIndex], snapShotsModel[comboSnapShots.currentIndex]);
                                 qmlMainWindow.startSingleCapture()
                            } else {
                                rowLayout4.isStartCapture = !rowLayout4.isStartCapture

                                comboBoxCamera.enabled = true
                                comboBoxLensType.enabled = true
                                comboW.enabled = true
                                comboH.enabled = true
                                comboSnapShots.enabled = true
                                calibrateMouseArea.visible = true

                                qmlMainWindow.stopSingleCapture()
                            }
                        }
                    }
                }
            }

            Item {
                id: calibrateItem
                Layout.fillWidth: true
                //Layout.fillHeight: true
                width: btnSize
                height: 50
                x: captureItem.x + captureItem.width + 80
                y: 5

                Rectangle {
                    id: calibrateHoverRect
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
                    id: calibrateText
                    color: "#ffffff"
                    text: "Calibrate"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent


                }

                Rectangle {
                    id: calibrateRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id:  calibrateMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        visible: false
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                calibrateHoverRect.visible = true;
                            }else{
                                calibrateHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            qmlMainWindow.setSingleParams(comboBoxCamera.currentIndex,comboBoxLensType.currentIndex, wModel[comboW.currentIndex],hModel[comboH.currentIndex], snapShotsModel[comboSnapShots.currentIndex]);

                            var isCalibSucess = qmlMainWindow.singleCalibrate();
                        }
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
     }

    function getCameraParams(curIndex) {
        if (curIndex < 0)
            curIndex = cameraCombo.currentIndex;
        cameraParamsSettingbox.getCameraParams(curIndex);
    }

    function updateConfiguration()
    {
        qmlMainWindow.updateCalibCameraView(calibCamView.camView)
    }

    function closeCalibSettingBox()
    {
        singleCalib.isVisibled = false
    }

    function setVisibleStopCaptureItem()
    {
        comboBoxCamera.enabled = true
        comboBoxLensType.enabled = true
        comboW.enabled = true
        comboH.enabled = true
        comboSnapShots.enabled = true
        calibrateMouseArea.visible = true
    }

    function setCaptureItem() {
        rowLayout4.isStartCapture = true
    }

    function setMessage(message) {
        messageText.text = message;
    }

    function setStrengthRatio(strengthRatio) {
        progress.value = strengthRatio * progress.maximumValue;
    }

    function setStrengthDrawColor(color) {

       if (color == "foundColor") {
           videoScreen.progressColor = "#ffffff";
       } else if (color == "stableColor") {
           videoScreen.progressColor = "#00ff00";
       } else  if (color == "normalColor") {
           videoScreen.progressColor = "#ffff00";
       }
    }

    function setCalibrateStatus(status) {
        calibrateMouseArea.visible = status
    }
}
