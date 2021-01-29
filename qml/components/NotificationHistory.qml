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
    id: notificationHistory

    property int        hoveredType: 0
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
    property int        nItemCount: 11
    property int        itemHeight:30
    property bool       isCPoint: false
    property int        textWidth: 45
    property int        lblWidth: 80
    property var        nfMsgList
    property string     imageUrl
    property string     contentLabel
    property string     typeLabel
    property string     timeLabel
    property bool       isOpen: false
    Rectangle {
        id: titleRectangle
        width: parent.width
        height: 48
        color: "#171717"
        z: 1

        Text {
            id: titleText
            anchors.left: parent.left
            anchors.leftMargin: 20
            y: (parent.height - height) / 2
            z: 3
            color: "#ffffff"
            text: qsTr("Notifications")
            font.bold: false
            font.pixelSize: 20
        }

        Item {
            id: clearItem
            x: (350 - width) / 2
            y: (parent.height - height) / 2
            z: 3
            width: 80
            height: 30
            anchors.right: parent.right


            Text {
                id: clearText
                x: (80 - width) / 2
                y: (parent.height - height) / 2
                z: 3
                color: "#429ce3"
                text: qsTr("Clear All")
                font.bold: false
                font.pixelSize: 14
            }

            Text {
                id: hoverText
                x: (80 - width) / 2
                y: (parent.height - height) / 2
                z: 3
                color: "#929292"
                text: qsTr("Clear All")
                font.bold: false
                font.pixelSize: 14
                visible: false
            }


            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                onEntered: {
                    hoverText.visible = true;
                    clearText.visible = false;
                }
                onExited: {
                    clearText.visible = true;
                    hoverText.visible = false;
                }
                onClicked: {
                    clearItem.visible = false;
                    notifyList.clear();
                    qmlMainWindow.removeNotification(-1);
                }
            }

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

//    Rectangle {
//        id: okRectangle
//        width: parent.width * 0.5
//        height: 40
//        color: "#0e3e64"
//        z: 1
//        anchors {
//            left: parent.left
//            bottom: parent.bottom
//        }

//        MouseArea {
//            anchors.fill: parent
//            onClicked: {
//                toolbox.clearSelected()
//                toolbox.setPlayAndPause();
//                notifyHistorybox.state = "collapsed"
//                isOpen = false;
//            }
//        }

//        Image {
//            id: okImage
//            x: (175 - width) / 2
//            y: (parent.height - height) / 2
//            width: 25
//            height: 25
//            fillMode: Image.PreserveAspectFit
//            source: "../../resources/btn_ok.PNG"
//        }
//    }

    Rectangle {
        id: cancelRectangle
        width: parent.width
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
                notifyHistorybox.state = "collapsed"
                isOpen = false;
            }
        }

        Image {
            id: cancelImage
            x: (350 - width) / 2
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
        height: parent.height
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

            ListView {
                id: notifyListView
                anchors.top: parent.top
                anchors.topMargin: 2
                width: parent.width
                height: scrollView.height - 40
                z: 1
               // spacing: 10

                model: ListModel {
                    id: notifyList

                }

                delegate: Item {
                    x: 5
                    width: parent.width
                    height: 49
                    Row {
                        width: parent.width
                        NotificationItem {
                            typeText: typeLabel
                            contentText: contentLabel
                            imagePath: imageUrl
                            timeText: timeLabel
                        }
                    }
                }

                remove: Transition {
                       ParallelAnimation {
                           //NumberAnimation { property: "opacity"; to: 0; duration: 1000 }
                           NumberAnimation { properties: "x"; to: 1000; duration: 300 }
                       }
                }

                removeDisplaced: Transition {
                       NumberAnimation { properties: "x,y"; duration: 1000 }
                   }
            }
        }
    }

    function setCalibSettings()
    {

    }

    function onDetails()
    {

    }

    function getNotifyHistory()
    {
        notifyList.clear();
        var nfCount = qmlMainWindow.getNotificationCount();
        if(nfCount === 0){
            clearItem.visible = false;
            return;
        }

        clearItem.visible = true;

        for(var i = 0; i < nfCount; i++)
        {
            var nfStr = qmlMainWindow.getNotification(i);
            nfMsgList = nfStr.split(":");
            parserMsg();
            notifyList.insert(0,{"imageUrl": imageUrl,"timeLabel": timeLabel
                                  ,"contentLabel": contentLabel})
        }

    }

    function parserMsg()
    {
        if (nfMsgList[0] === "Error")
        {
            imageUrl = "../../resources/ico_error.png"
           // typeLabel = "Error"
        }
        else if(nfMsgList[0] === "Warning")
        {
            imageUrl = "../../resources/ico_warning.png"
            //typeLabel = "Warning"
        }
        else if(nfMsgList[0] === "Information")
        {
            imageUrl = "../../resources/ico_notify.png"
            //typeLabel = "Information"
        }

        timeLabel = nfMsgList[2] + ":" + nfMsgList[3];
        contentLabel = nfMsgList[1];
    }

    function hiddenNotification(){
          notificationHistory.x = root.width
        notificationHistory.state = "collapsed";
    }

    function closeBox()
    {
        toolbox.clearSelected();
        statusItem.setPlayAndPause();
        notifyHistorybox.state = "collapsed"
        isOpen = false;
    }

}
