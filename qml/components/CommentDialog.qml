import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import "../controls"

ToolWindow{
    width: 300
    height: 140
    z: 10
    windowState: qsTr("windowed")
    visible: false
    floatingWindowBorderColor: "#1f1f1f"
    property int    fontSize: 15

    property int        leftMargin: 20
    property int        rightMargin: 20
    property int        spacing: 20
    property color      textColor: "#ffffff"
    property int        lblFont: 14
    property int        itemHeight: 30
    property int        curIndex
    property string     cameraName
    property string     title
    property int        lblWidth: 40

    handle: Rectangle {
        color: "#1f1f1f"
        anchors.margins: 4
        anchors.centerIn: parent.Center
        height: 30

        Text {
            y: (30 - height) / 2
            color: "#ffffff"
            text: "Please leave a comment"
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.left: parent.left
            textFormat: Text.PlainText
            wrapMode: Text.WordWrap
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: fontSize
        }
    }

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
            x: 0
            y: 0
            width: scrollView.width
            height: scrollView.height

            Item {
                id: commentItem
                width: parent.width
                height: itemHeight
                anchors.topMargin: spacing
                anchors.top: parent.top

                FlatText {
                    id: commentStr
                    anchors.right: parent.right
                    anchors.rightMargin: spacing
                    width: parent.width - spacing * 2
                    height: itemHeight
                    anchors.verticalCenter: parent.verticalCenter
                    placeholderText: qsTr("Please leave a comment")                    

                    Keys.onPressed: {
                        if (event.key == Qt.Key_Enter || event.key == Qt.Key_Return)
                            onCheckDialog();
                        else if (event.key == Qt.Key_Escape)
                            onCloseDialog();
                    }
                }
            }

            Rectangle {
                id: okRectangle
                width: parent.width * 0.5
                height: 35
                color: "#0e3e64"
                opacity: 1
                z: 1
                anchors {
                    left: parent.left
                    bottom: parent.bottom
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: onCheckDialog()
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
                height: 35
                color: "#1f1f1f"
                opacity: 1
                z: 1
                anchors {
                    right: parent.right
                    bottom: parent.bottom
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: onCloseDialog()
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

    function onCheckDialog(){
        onCloseDialog();
    }

    function onCloseDialog(){
		statusItem.stopRecordTake(commentStr.text);        
        commentStr.text = "";
        statusItem.setPlayMode();
    }

    function setFocus(){
    }

}
