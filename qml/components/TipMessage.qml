import QtQuick 2.0

Rectangle {
    id: tipMessageRectangle
    x: 0
    y: 0
    z: 14
    width: 82
    height: 32
    color: "#000000"
    radius: 1
    visible: false

    property string      toolTip: ""
    property color       tipMSGColor: "#ffffff"
    property int         fontSize: 0

    Text {
        id: tipText
        x: 4
        y: 5
        z: 14
        color: "#ffffff"
        text: toolTip
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: 19
    }
}
