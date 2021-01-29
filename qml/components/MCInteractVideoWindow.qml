import QtQuick 2.4
import "../controls"

FloatingInteractiveWindow  {
    property int uiSpacing: 4
    property bool active
    property bool destroyOnClose: false
    property  int  toolwindowHandleHeight: 0
    id: root
    width: 350
    height: 350
    windowMenuGroup: 'video'
    //minimizedWidth: 7 * uiSpacing + img0.width + img1.width + img2.width //+ img3.width
    maximizable: true
    minimizable: true
    signal closing()
    /*handle: Rectangle {
        height: 30

        /*gradient: Gradient {
            GradientStop { position: 0.0; color: "#ffffff" }
            GradientStop { position: 0.29; color: "#404040" }
            GradientStop { position: 1.0; color: "#434343" }
        }*/
       /* color: "#000000"
        Image {
            id: img0
            anchors.left: parent.left
            anchors.leftMargin: uiSpacing
            anchors.verticalCenter: parent.verticalCenter
           // source: active ? "resouces/icon-green-circle.png": "/resources/icon-red-circle.png"
            width: parent.height * 0.5
            height: parent.height * 0.5
        }

		/*
        MCBasicText {
            wrapMode: "NoWrap"

            id: captText
            color: _S.uiMainFontColor
            anchors.left: img0.right
            anchors.right: r1.left
            clip: true
            anchors.verticalCenter: parent.verticalCenter
            text: "Mova Camera"
            anchors.leftMargin: uiSpacing
        }
		*/
        /*clip: true

        Row {
            id: r1
            anchors.verticalCenter: parent.verticalCenter
            anchors.right: parent.right
            anchors.rightMargin: uiSpacing
            spacing: uiSpacing
            Image {
                id: img1
                source: "/resources/icon-minimize.png"
                width:  handle.height * 0.7
                height: handle.height * 0.7
                anchors.verticalCenter: parent.verticalCenter
                MouseArea {
                    anchors.fill: parent
                    onClicked: toggleMinimized()
                }
            }
            Image {
                id: img2
                source: "/resources/icon-expand-black.png"
                width: handle.height * 0.7
                height: handle.height * 0.7
                anchors.verticalCenter: parent.verticalCenter
                MouseArea {
                    anchors.fill: parent
                    onClicked: toggleMaximized()
                }
            }
            Image {
                id: img3
                source: "/resources/icon-close-black.png"
                width: handle.height * 0.7
                height: handle.height * 0.7
                anchors.verticalCenter: parent.verticalCenter
                MouseArea {
                    anchors.fill: parent
                    onClicked: {

                        root.hidden = true
                        if(destroyOnClose) {
                            close();
                        }
                    }
                }
            }
        }


    }*/
//    Rectangle {
//        //TODO: fast video surface should be here
//    }
}

