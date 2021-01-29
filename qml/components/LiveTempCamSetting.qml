import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import "../controls"

Item {
    id: item1
    width : 250
    height: 300
    opacity: 1


    property int hoveredType: 0
    property bool isHovered : false
    property bool isSelected: false
    property int  leftMargin: 20
    property int rightMargin: 20
    property int spacing: 20
    property color textColor: "#ffffff"
    property color comboTextColor: "#7899ff"
    property color comboBackColor: "#343434"
    property color spliterColor: "#555555"
    property int lblFont: 14
    property int groupFont: 16
    property int nItemCount: 6
    property int itemHeight:30
    property int stereoType: 0
    property int curIndex
    property string cameraName
    property string title


    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#171717"
        border.color: "#1f1f1f"
        border.width: 1
        opacity: 0.9

    }




    ScrollView {
        id: scrollView
        width: parent.width
        height: parent.height
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
            height: scrollView.height


            Item {
                id: titleItem
                width: parent.width
                height: 30
                anchors.top: parent.top
                Text {
                    id:  title
                    x: (parent.width - width) / 2
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#ffffff"
                    text: qsTr("")
                    font.pixelSize: 15
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }


            }

            Spliter {
                width: parent.width - 4
                x: ( parent.width - width ) / 2
                anchors.top: titleItem.bottom
            }




            Item {
                id: bothItem
                width: parent.width
                height: 30
                anchors.top: titleItem.bottom
                anchors.topMargin: spacing
                Text {
                    id:  bothText
                    x: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#ffffff"
                    text: qsTr("Both")
                    font.pixelSize: lblFont
                }

                Switch{
                    id: bothSwitch
                    x: ( parent.width - width ) / 2
                    width: 50
                    height: 30
                    //checked: true
                    onCheckedChanged: {
                        if(bothSwitch.checked) {

                            stereoType = 3
                            leftRightSwitch.enabled = false;
                        }else {
                            if(leftRightSwitch.checked){
                                stereoType = 2;
                            } else {
                                stereoType =1;
                            }

                            leftRightSwitch.enabled = true;
                            leftRightSwitch.checked;
                        }
                    }
                }
            }

            Item {
                id: leftRightItem
                width: parent.width
                height: 30
                anchors.top: bothItem.bottom
                anchors.topMargin: spacing
                Text {
                    id:  leftRight
                    x: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#ffffff"
                    text: qsTr("Left")
                    font.pixelSize: lblFont
                }

                Switch{
                    id: leftRightSwitch
                    x: ( parent.width - width ) / 2
                    width: 50
                    height: 30
                   // checked: false
                    //enabled: false
                    onCheckedChanged: {
                        if(leftRightSwitch.checked){
                            stereoType = 2;
                        }else{
                            stereoType = 1;
                        }
                    }
                }
                Text {
                    id:  right
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#ffffff"
                    text: qsTr("Right")
                    font.pixelSize: lblFont
                }
            }


            Item {
                id: okItem
                width: 65
                anchors.bottom: parent.bottom
                anchors.bottomMargin: spacing - 10
                anchors.right: cancelItem.left
                anchors.rightMargin: 5
                height: 30

                Rectangle {
                    id: okHoverRect
                    x: 0
                    width: parent.width
                    height: parent.height
                    anchors.fill: parent
                    color: "#353535"
                    border.color: "#4e8d15"
                    border.width: 1
                    visible: false
                    //radius: 4
                    z: 1
                }


                Text {
                    id: okText
                    color: "#ffffff"
                    text: "OK"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent


                }

                Rectangle {
                    id: okRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#353535"
                  //  radius: 4


                    MouseArea {
                        id: calcultorMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                okHoverRect.visible = true;
                            }else{
                                okHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            qmlMainWindow.setTempStereoType(curIndex, stereoType);
                            liveTempCamSetting.state = "collapsed";
                        }
                    }
                }
            }

            Item {
                id: cancelItem
                width: 65
                anchors.bottom: parent.bottom
                anchors.bottomMargin: spacing  -10
                height: 30
                anchors.right: parent.right
                anchors.rightMargin: rightMargin - 10

                Rectangle {
                    id: cancelHoverRect
                    x: 0
                    width: parent.width
                    height: parent.height
                    anchors.fill: parent
                    color: "#353535"
                    visible: false
                    border.color: "#4e8d15"
                    border.width: 1
                    z: 1
                }


                Text {
                    id: cancelText
                    color: "#ffffff"
                    text: "Cancel"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent


                }

                Rectangle {
                    id: cancelRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#353535"

                    MouseArea {
                        id: cancelMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                cancelHoverRect.visible = true;
                            }else{
                                cancelHoverRect.visible = false;
                            }
                        }
                        onClicked: {
                            liveTempCamSetting.state = "collapsed";
                        }

                    }
                }
            }
         }
     }

    function getCameraValues(videoIndex){
        curIndex = videoIndex;
        var type =  qmlMainWindow.getTempStereoType(curIndex)
        switch(type){
        case 1:
            leftRightSwitch.checked = false;
            bothSwitch.checked = false;
            break;
        case 2:
             leftRightSwitch.checked = true;
            bothSwitch.checked = false;
            break;
        case 3:
            bothSwitch.checked = true;
            leftRightSwitch.enabled = false;
            break;
        default:
            break;
        }
    }

    function setCameraValues(videoIndex)
    {
        curIndex = videoIndex;
        var item = camList.get(curIndex);
        cameraName = item.titleText;
        var pos = cameraName.lastIndexOf("/") + 1;
        title.text = cameraName.substring(pos);

    }

}
