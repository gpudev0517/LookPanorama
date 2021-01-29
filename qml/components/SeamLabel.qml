import QtQuick 2.0

Item {
    property int    seamIndex: -1
    property int    xPos: 0
    property int    yPos: 0

    Rectangle {
        id: seamLabel
        x: xPos
        y: yPos
        width: 30
        height: 20
        radius: 4
        border.color: "#000000"
        color: "#4e8d15"
        border.width: 1

        Text {
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            text: seamIndex
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            color: "white"
            anchors.verticalCenter: parent.verticalCenter
        }
    }
}

