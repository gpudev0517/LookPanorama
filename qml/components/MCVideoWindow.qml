import QtQuick 2.4
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4

import "../controls"

FloatingWindow {
    property int    uiSpacing: 4
    property bool   active
    property bool   destroyOnClose: false
    property  int   toolwindowHandleHeight: 50
    property bool   isClicked: false
    property bool   isDblClicked: false
    property int    preX
    property int    preY
    id: root
    windowMenuGroup: 'video'
    maximizable: true
    minimizable: true
    signal closing()
    onWidthChanged: resetCPoint()
    //onXChanged: resetCPoint();
    handle: Rectangle {
        height: 30
        color: "#000000"
        Image {
            id: img0
            anchors.left: parent.left
            anchors.leftMargin: uiSpacing
            anchors.verticalCenter: parent.verticalCenter
//            source: "/resources/icon-minimize.png"
            width: parent.height * 0.5
            height: parent.height * 0.5
        }


        Label {
            wrapMode: "NoWrap"
            id: captText
            color: "#ffffff"
            anchors.left: img0.right
            clip: true
            anchors.verticalCenter: parent.verticalCenter
            text: caption
            anchors.leftMargin: 4
            font.pointSize: 8
            font.bold: true
        }

        clip: true

        Row {
            id: r1
            anchors.verticalCenter: parent.verticalCenter
            anchors.right: parent.right
            anchors.rightMargin: uiSpacing
            spacing: uiSpacing
            z: 2

//            Image {
//                id: settingImage
//                source: "/resources/setting.png"
//                width: handle.height * 0.7
//                height: handle.height * 0.7
//                anchors.verticalCenter: parent.verticalCenter
//                MouseArea {
//                    anchors.fill: parent
//                    onClicked: setCameraSetting();
//                }
//            }
            Image {
                id: expandImage
                source: "/resources/icon-expand-black.png"
                width: 23
                height: 23
                fillMode: Image.PreserveAspectFit
                anchors.verticalCenter: parent.verticalCenter

//                Rectangle {
//                    id: expandHoverRetangle
//                    z: -1
//                    width: 30
//                    height: 30
//                    color: "#1f1f1f"
//                    visible: false
//                }

                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: {
                        cameraView.expandWindow();
                    }
                    onEntered: expandHoverRetangle.visible = true;
                    onExited: expandHoverRetangle.visible = false;
                }
            }
        }


        Rectangle {
            id: expandHoverRetangle
            z: 1
            width: 30
            height: 30
            anchors.right: parent.right
            color: "#1f1f1f"
            visible: false
        }

        MouseArea {
            anchors.fill: parent
            onDoubleClicked: {
                cameraView.expandWindow();
            }
        }

        Timer {
            id: cPointTimer
            interval: 50
            running: false
            repeat: false
            onTriggered: resetCPoint();
        }
   }

    function resetCPoint()
    {
        if(!liveBox.isCPoint) return;
        //clearCPoint();
//        if(isDblClicked)
//        {
//            frameWidth = centralItem.width;
//            frameHeight = centralItem.height - 30;
//        }
//        else {
//            frameWidth = 400;
//            frameHeight = 300;
//        }
        if(cameraView.width === 400 || cameraView.width === centralItem.width)
        {
            clearCPoint();
            createCPoint();
        }
        liveGridView.currentIndex = deviceNum;
    }

}

