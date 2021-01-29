import QtQuick 2.5

Item {
    width : 250
    height: 684
    opacity: 1


    property int hoveredType: 0
    property bool isHovered : false
    property bool isSelected: false

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
            x: (250 - width) / 2
            y: (parent.height - height) / 2
            z: 3
            color: "#ffffff"
            text: qsTr("Help")
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
                toolbox.clearSelected()
                aboutBox.state = "collapsed"
            }
        }

        Image {
            id: okImage
            x: (125 - width) / 2
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
                aboutBox.state = "collapsed"
            }
        }

        Image {
            id: cancelImage
            x: (125 - width) / 2
            y: (parent.height - height) / 2
            width: 25
            height: 25
            fillMode: Image.PreserveAspectFit
            source: "../../resources/btn_cancel.PNG"
        }
    }

}
