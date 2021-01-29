import QtQuick 2.0

Item {
    width: 360
    height: 80

    property string     typeText: "Error"
    property string     contentText: "Terminating audio and camera capture threads..."
    property string     imagePath: "../../resources/ico_error.png"

    Rectangle {
        id: background
        width: parent.width
        height: parent.height
        color: "#171717"

    }

    Item {
        id: typeItem
        width: 50
        height: 50

        Image {
            anchors.centerIn: parent
            width: 40
            height: 40
            source: imagePath
            fillMode: Image.PreserveAspectFit
        }
    }

    Item {
        id: titleItem
        width: parent.width - typeItem.width
        height: parent.height / 3
        anchors.left: typeItem.right

        Text {
            id: title
            width: 100
            height: 27
            color: "#ffffff"
            text: typeText
            font.pointSize: 12
            font.bold: true
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
        }

        Rectangle {
            width: parent.width / 8
            height: parent.height
            anchors.right: parent.right
            color: "#171717"

            Image {
                width: parent.width
                height: parent.height
                source: "../../resources/icon-close-black.png"
                fillMode: Image.PreserveAspectFit
                anchors.centerIn: parent

                Rectangle {
                    id: closeHoverRetangle
                    z: -1
                    width: parent.width
                    height: parent.height
                    color: "#c75050"
                    visible: false
                }

            }

            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                onEntered: closeHoverRetangle.visible = true;
                onExited: closeHoverRetangle.visible = false;

                onClicked: {
                    notification.state = "collapsed";

                }
            }
        }
    }

    Item {
        id: contentItem
        width: parent.width - typeItem.width
        height: parent.height * 2 / 3
        anchors.left: typeItem.right
        anchors.top: titleItem.bottom

        Text {
            id: content
            width: parent.width
            color: "#929292"
            text: contentText
            wrapMode: Text.WordWrap
            textFormat: Text.RichText
            font.pointSize: 12
            //anchors.fill: parent
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 15
        }
    }
}
