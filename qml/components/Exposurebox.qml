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
    height: 684
    opacity: 1


    property int hoveredType: 0
    property bool isHovered : false
    property bool isSelected: false
    property int  leftMargin: 20
    property int rightMargin: 20
    property int spacing: 15
    property color textColor: "#ffffff"
    property color comboTextColor: "#7899ff"
    property color comboBackColor: "#343434"
    property color spliterColor: "#555555"
    property int lblFont: 14




    Rectangle {
        id: titleRectangle
        x: 0
        y: 0
        width: parent.width
        height: 48
        color: "#171717"
        z: 1
        opacity: 1

        Text {
            id: titleText
            x: (350 - width) / 2
            y: (parent.height - height) / 2
            z: 3
            color: "#ffffff"
            text: qsTr("Exposure")
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
        opacity: 1
        z: 1
        anchors {
            left: parent.left
            bottom: parent.bottom
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                qmlMainWindow.onOkGain();
                toolbox.clearSelected()
                statusItem.setPlayAndPause();
                exposureBox.state = "collapsed"
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
        opacity: 1
        z: 1
        anchors {
            right: parent.right
            bottom: parent.bottom
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                toolbox.clearSelected()
                statusItem.setPlayAndPause();
                exposureBox.state = "collapsed"
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
        opacity: 0.8
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
            x: 0
            y: 0
            width: scrollView.width
            //height:root.height - titleRectangle.height - okRectangle.height
            height: 684 - titleRectangle.height


            ListView {
                id: previewItem
                x: 30
                anchors.top: parent.top
                anchors.topMargin: spacing + 20
                width: 300
                height: 250

            }

            Item {
                id: calculatorItem
                width: 65
                anchors.top: previewItem.bottom
                anchors.topMargin: spacing
                anchors.right: resetItem.left
                anchors.rightMargin: 10
                height: 30

                Rectangle {
                    id: calculatorHoverRect
                    x: 0
                    width: parent.width
                    height: parent.height
                    anchors.fill: parent
                    color: "#373737"
                    border.color: "#4e8d15"
                    border.width: 1
                    visible: false
                    z: 1
                }


                Text {
                    id: calculatorText
                    color: "#ffffff"
                    text: "Calculator"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent


                }

                Rectangle {
                    id: calculatorRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id: calcultorMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                calculatorHoverRect.visible = true;
                            }else{
                                calculatorHoverRect.visible = false;
                            }
                        }

                        onClicked: qmlMainWindow.onCalculatorGain();

                    }
                }
            }

            Item {
                id: resetItem
                width: 65
                anchors.top: previewItem.bottom
                anchors.topMargin: spacing
                anchors.right: parent.right
                height: 30
                anchors.rightMargin: rightMargin

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
                    color: "#ffffff"
                    text: "Reset"
                    z: 1
                    font.pointSize: 10
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

                        onClicked: qmlMainWindow.onResetGain();

                    }
                }
            }
        }
    }
}
