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




            /*Item {
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
            }*/

            Item {
                id: noneItem
                width: parent.width
                height: 30
                anchors.top: titleItem.bottom
                anchors.topMargin: spacing

                RadioButton {
                    id: noneRadio
                   // x: ( parent.width - width ) / 2 - 20
                    z: 1
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    width: 120
                    height: 30
                    //checked: true
                    onCheckedChanged: {
                        if(noneRadio.checked) {
                            stereoType = 0;
                            bothRadio.checked = false;
                            leftRadio.checked = false;
                            rightRadio.checked = false;
                        }
                    }
                }

                Text {
                    id:  noneLabel
                    x: 50
                    z: 0
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#ffffff"
                    text: qsTr("None")
                    font.pixelSize: lblFont
                }
            }

            Item {
                id: bothItem
                width: parent.width
                height: 30
                anchors.top: noneItem.bottom
                anchors.topMargin: spacing / 2

                RadioButton {
                    id: bothRadio
                    z: 1
                   // x: ( parent.width - width ) / 2 - 20
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    width: 100
                    height: 30
                    //checked: true
                    onCheckedChanged: {
                        if(bothRadio.checked) {
                            stereoType = 3;
                            noneRadio.checked = false;
                            leftRadio.checked = false;
                            rightRadio.checked = false;
                        }
                    }
                }

                Text {
                    id:  bothLabel
                    x: 50
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#ffffff"
                    text: qsTr("Both")
                    z: 0
                    font.pixelSize: lblFont
                }
            }

            Item {
                id: leftItem
                width: parent.width
                height: 30
                anchors.top: bothItem.bottom
                anchors.topMargin: spacing / 2

                RadioButton {
                    id: leftRadio
                    z: 1
                   // x: ( parent.width - width ) / 2 - 20
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    width: 100
                    height: 30
                    //checked: true
                    onCheckedChanged: {
                        if(leftRadio.checked) {
                            stereoType = 1;
                            noneRadio.checked = false;
                            bothRadio.checked = false;
                            rightRadio.checked = false;
                        }
                    }
                }

                Text {
                    id:  leftLabel
                    x: 50
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#ffffff"
                    text: qsTr("Left")
                    font.pixelSize: lblFont
                }
            }

            Item {
                id: rightItem
                width: parent.width
                height: 30
                anchors.top: leftItem.bottom
                anchors.topMargin: spacing / 2

                RadioButton {
                    id: rightRadio
                   // x: ( parent.width - width ) / 2 - 20
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    width: 100
                    height: 30
                    //checked: true
                    onCheckedChanged: {
                        if( rightRadio.checked ) {
                            stereoType = 2;
                            noneRadio.checked = false;
                            bothRadio.checked = false;
                            leftRadio.checked = false;
                        }
                    }
                }

                Text {
                    id:  rightLabel
                    x: 50
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#ffffff"
                    text: qsTr("Right")
                    font.pixelSize: lblFont
                }
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
                        bCamFlag = false;
                        qmlMainWindow.setTempStereoType(curIndex, stereoType)
                        cameraSettingsbox.state = "collapsed";
                    }
                }

                Image {
                    id: okImage
                    x: (parent.width - width) / 2
                    y: (parent.height - height) / 2
                    width: 25
                    height: 25
                    fillMode: Image.PreserveAspectFit
                    source: "../../resources/check.png"
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
                        bCamFlag = false;
                        cameraSettingsbox.state = "collapsed";
                    }
                }

                Image {
                    id: cancelImage
                    x: (parent.width - width) / 2
                    y: (parent.height - height) / 2
                    width: 25
                    height: 25
                    fillMode: Image.PreserveAspectFit
                    source: "../../resources/uncheck.png"
                }
            }
         }
       }



    function getCameraValues(videoIndex){
        curIndex = videoIndex;
        var type =  qmlMainWindow.getTempStereoType(curIndex)
        switch(type){
        case 0:
            noneRadio.checked = true;
            bothRadio.checked = false;
            leftRadio.checked = false;
            rightRadio.checked = false;
            break;
        case 1:
            noneRadio.checked = false;
            bothRadio.checked = false;
            leftRadio.checked = true;
            rightRadio.checked = false;
            break;
        case 2:
            noneRadio.checked = false;
            bothRadio.checked = false;
            leftRadio.checked = false;
            rightRadio.checked = true;
            break;
        case 3:
            noneRadio.checked = false;
            bothRadio.checked = true;
            leftRadio.checked = false;
            rightRadio.checked = false;
            break;
        default:
            break;
        }
    }

    function setCameraValues(videoIndex)
    {
        curIndex = videoIndex;
        var item = videoList.get(curIndex);
        cameraName = item.titleText;
        var pos = cameraName.lastIndexOf("/") + 1;
        title.text = cameraName.substring(pos);


    }

}
