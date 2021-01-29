import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import "../controls"

Item {
    width : 350
    height: 700

    property int        hoveredType: 0
    property bool       isHovered : false
    property bool       isSelected: false
    property int        leftMargin: 20
    property int        rightMargin: 20
    property int        spacing: 20
    property color      textColor: "#ffffff"
    property color      comboTextColor: "#7899ff"
    property color      comboBackColor: "#343434"
    property color      spliterColor: "#555555"
    property int        lblFont: 14
    property int        groupFont: 16
    property int        nItemCount: 20
    property int        itemHeight:30
    property bool       isCPoint: false
    property int        textWidth: 45
    property int        lblWidth: 80
    property bool       isDetailed: false

    Rectangle {
        id: titleRectangle
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
            text: qsTr("Live")
            font.bold: false
            font.pixelSize: 20
        }
    }

    Spliter {
        id: spliterRectangle
        width: parent.width
        height: 2
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
                toolbox.clearSelected()
                statusItem.setPlayAndPause();
                setCalibSettings();
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
                toolbox.clearSelected();
                statusItem.setPlayAndPause();
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
            transientScrollBars: true
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
            height: (spacing + itemHeight )* nItemCount + 80


            Item {
                id: camListItem
                y: 0
                width: 350
                height: 30
                anchors.top: parent.top
                anchors.topMargin: spacing

                Text {
                    id: camListText
                    y: 12
                    color: textColor
                    text: qsTr("Calibration")
                    font.pixelSize: groupFont
                    font.bold: true
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                }
            }

            Spliter {
                id: calibrateSpliter
                width:  parent.width
                anchors.top: camListItem.bottom
                anchors.topMargin: 10
                anchors.left: parent.left
                anchors.leftMargin: 2
            }

            Item {
                id: autoCalib
                width: parent.width
                height: 30
                anchors.top: calibrateSpliter.bottom
                anchors.topMargin: itemHeight
                Text {
                    id: lensTypeLabel
                    width: lblWidth
                    color: textColor
                    text: qsTr("Lens Type")
                    horizontalAlignment: Text.AlignLeft
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    font.pixelSize: lblFont

                }

                ComboBox {
                    id: lensTypeCombo
                    width: parent.width / 4
                    height: 30
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.left: lensTypeLabel.right
                    anchors.leftMargin: spacing
                    model: ["Standard", "Fullframe Fisheye", "Circular Fisheye", "CV Standard", "CV Fisheye"]
                    onCurrentIndexChanged:
                    {
                        if (currentIndex)
                            hfov.text = 240;
                        else
                            hfov.text = 120;
                    }
                }

                Text {
                    id: hfovLabel
                    color: textColor
                    text: qsTr("HFOV")
                    horizontalAlignment: Text.AlignLeft
                    anchors.left: lensTypeCombo.right
                    anchors.leftMargin: 10
                    anchors.verticalCenter: parent.verticalCenter
                    font.pixelSize: lblFont

                }

                FlatText {
                    id: hfov
                    width: textWidth
                    height: 30
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                }
            }

            Item {
                id: detailItem
                width: 65
                anchors.top: autoCalib.bottom
                height: 30
                anchors.topMargin: 10
                anchors.right: calibrateItem.left
                anchors.rightMargin: rightMargin

                Rectangle {
                    id: detailHoverRect
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
                    id: detailText
                    color: "#ffffff"
                    text: "Details"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent


                }

                Rectangle {
                    id: detailRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id: detailMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                detailHoverRect.visible = true;
                            }else{
                                detailHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            onDetails();
                        }
                    }
                }
            }


            Item {
                id: calibrateItem
                width: 65
                anchors.top: autoCalib.bottom
                height: 30
                anchors.topMargin: 10
                anchors.right: parent.right
                anchors.rightMargin: rightMargin

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
                        id: calibrateMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                calibrateHoverRect.visible = true;
                            }else{
                                calibrateHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            root.onCalibrate();
                            setCalibSettings();
                            statusItem.setPauseMode();
                            qmlMainWindow.startCalibrating();
                        }

                    }
                }
            }


            Item {
                id: cPointItem
                y: 0
                width: 350
                height: 30
                anchors.top: calibrateItem.bottom
                anchors.topMargin: spacing

                Text {
                    id: threeModeText
                    y: 12
                    color: textColor
                    text: qsTr("View Control Points")
                    font.pixelSize: lblFont
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                }

                Switch {
                    id: controlPointSwitch
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    anchors.verticalCenter: parent.verticalCenter
                    width:parent.width / 2 -20
                    height: 30
                    onClicked: {
                        if(controlPointSwitch.checked)
                        {
                            isCPoint = true;
                            liveView.createCPoint();
                        }
                        else
                        {
                            isCPoint = false;
                            liveView.clearCPoint();
                        }

                    }

                }
            }
        }
    }

    function setCalibSettings()
    {
        // Auto Calibration Parameters
        qmlMainWindow.setFov(hfov.text);
        qmlMainWindow.setLensType(lensTypeCombo.currentIndex);
    }

    function onDetails()
    {
        if(isDetailed) return;
        liveView.createDetailsWindow();
    }

    function getLensSettings()
    {
        //hfov.text = qmlMainWindow.getFov();
        lensTypeCombo.currentIndex = qmlMainWindow.getLensType();
    }

    function initControlPoint(){
        controlPointSwitch.checked = false;
        isCPoint = false;
        liveView.clearCPoint();
    }
}
