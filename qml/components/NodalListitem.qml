import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4

Item{
    width:  root.width
    height: 30

    property string          videofilePath: "title"
    property string          imagefilePath: ""
    property string          titleTextColor
    property bool            checkSelect: false

    Rectangle {
        id: hoveredRectangle
        x: 20
        y: -5
        width: parent.width - 50
        height: parent.height + 10
        color: "#1f1f1f"
        visible: false

    }
    Rectangle {
        id: selectRectangle
        x: 20
        y: -5
        width: parent.width - 50
        height: parent.height + 10
        color: "#0e3e64"
        visible: checkSelect
    }

    Image {
        id: videoIcon
        width: 30
        height: 30
        source: "../../resources/icon_video_small.png"
        x: 50
        y: 0
        z: 1
        fillMode: Image.PreserveAspectFit
    }

    Text {
        id: videofilePathLabel
        x: videoIcon.x + videoIcon.width + 20
        y: (parent.height - height) / 2
        z: 1
        //color: titleTextColor
        color: nodalListView.enableFootage ? "#ffffff": "#8a8a8a"
        text: qsTr(videofilePath)
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        font.pixelSize: 15
    }

    MouseArea {
        id: videoPathMouseArea
        x: videoIcon.x
        width: videoIcon.width + videofilePathLabel.width + 20
        height: parent.height
        z: 2
        hoverEnabled: true
        enabled: nodalListView.enableFootage ? true: false
        onClicked: {
            nodalShootvideoDialog.open();
        }
    }

    Image {
        id: videoclearImage
        width: 30
        height: 30
        source: "../../resources/uncheck.png"
        anchors.right: spliterRectangle.left
        anchors.rightMargin: 10
        y: 0
        z: 1
        fillMode: Image.PreserveAspectFit

        MouseArea {
            id: videoclearMouseArea
            anchors.fill: parent
            z: 2
            hoverEnabled: true
            enabled: nodalListView.enableFootage ? true: false
            onClicked: {
                nodalListView.initfootageVideo();
            }
        }
    }


    Rectangle {
        id: spliterRectangle
        width: 2
        height: parent.height
        color: "#8a8a8a"
        x: parent.width / 2
    }

    Image {
        id: imageIcon
        width: 30
        height: 30
        source: "../../resources/icon_image_small.png"
        anchors.left: spliterRectangle.right
        anchors.leftMargin: 10
        z: 1
        fillMode: Image.PreserveAspectFit
    }

    Text {
        id: imagefilePathLabel
        x: imageIcon.x + imageIcon.width + 25
        y: (parent.height - height) / 2
        z: 1
        //color: titleTextColor
        color: nodalListView.enableFootage ? "#ffffff": "#8a8a8a"
        text: qsTr(imagefilePath)
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        font.pixelSize: 15
    }

    MouseArea {
        id: imagePathMouseArea
        x: imageIcon.x
        width: imageIcon.width + imagefilePathLabel.width + 25
        height: parent.height
        z: 2
        hoverEnabled: true
        enabled: nodalListView.enableFootage ? true: false
        onClicked: {
            nodalShootimageDialog.open();
        }
    }

    Image {
        width: 30
        height: 30
        source: "../../resources/uncheck.png"
        anchors.right: parent.right
        anchors.rightMargin: 50
        z: 1
        fillMode: Image.PreserveAspectFit

        MouseArea {
            id: imageclearMouseArea
            anchors.fill: parent
            z: 2
            hoverEnabled: true
            enabled: nodalListView.enableFootage ? true: false
            onClicked: {
                nodalListView.initfootageImage();
            }
        }
    }
}
