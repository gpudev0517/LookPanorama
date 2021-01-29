import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2

import "../controls"

Item {
    width : 350
    height: 684

    property bool isHovered : false
    property bool isSelected: false
    property int  leftMargin: 20
    property int rightMargin: 20
    property int spacing: 15
    property color textColor: "#ffffff"

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
            text: qsTr("Snapshot")
            font.bold: false
            font.pixelSize: 20
        }
    }

    Rectangle {
        id: spliterRectangle
        width: parent.width
        height: 2
        z: 3
        anchors.top: titleRectangle.bottom
        color: "#1f1f1f"
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
                //onBackMode()
                qmlMainWindow.setSnapshotDir(savePathFeild.text);
                snapshotBox.state = "collapsed"
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
                //onBackMode()
                snapshotBox.state = "collapsed"
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
    function onBackMode(){
        switch(root.curMode){
        case 1:
            liveView.state = "show"
            break;
        case 2:
            sphericalImage.state = "show"
            break;
        case 3:
            interactiveImage.state = "show"
            break;
        default:
            break;
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
                //implicitHeight: leftMargin
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
            //handleOverlap: 70

        }


        Item{
            id: groupItems
            width: parent.width
            //height:root.height - titleRectangle.height - okRectangle.height
            height: 100

            Item {
                id: savePathItem
                width: 350
                height: 30
                anchors.top: parent.top
                anchors.topMargin: spacing + 20
                Text {
                    id: savePathText
                    width: 100
                    y: 12
                    color: "#ffffff"
                    text: qsTr("Save Path")
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    font.pixelSize: 14

                }

                FlatText {
                    id: savePathFeild
                    width: parent.width * 0.5
                    height: 30
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.left: savePathText.right
                    anchors.rightMargin: rightMargin
                }

                Rectangle{
                    width: 30
                    height: 30
                    z: 1
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#373737"
                    //border.color: "#4e8d15"
                    Text{
                        anchors.fill: parent
                        text: "  ..."
                        color: "#4e8d15"
                        verticalAlignment: Text.AlignVCenter
                        z: 3
                    }

                }
                Rectangle{
                    id: fileHoveredRectangle
                    width: 30
                    height: 30
                    z: 1
                    color: "#373737"
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    anchors.verticalCenter: parent.verticalCenter
                    border.color: "#4e8d15"
                    border.width: 1
                    Text{
                        anchors.fill: parent
                        text: "  ..."
                        color: "#4e8d15"
                        verticalAlignment: Text.AlignVCenter
                        z: 3
                    }

                    visible: false
                }
                MouseArea{
                    x: fileHoveredRectangle.x
                    z: 2
                    width: fileHoveredRectangle.width
                    height: fileHoveredRectangle.height
                    anchors.verticalCenter: parent.verticalCenter
                    hoverEnabled: true
                    onEntered: fileHoveredRectangle.visible = true
                    onExited: fileHoveredRectangle.visible = false
                    onClicked: saveFileDialoge.visible = true
                }
            }

            FileDialog{
                id: saveFileDialoge
                title: "Select configuration file"
                selectMultiple: false
                selectFolder: true;

                onSelectionAccepted: {
                    var fileName = fileUrl.toString().substring(8); // Remove "file:///" prefix
                    savePathFeild.text = fileName;
                }
            }
        }
    }

    function getSnapshotDir()
    {
        savePathFeild.text = qmlMainWindow.getSnapshotDir();
    }
}
