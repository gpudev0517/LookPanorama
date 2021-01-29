import QtQuick 2.5
import QtQuick.Layouts 1.1
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick 2.5
import "../../resources"
import "../components"
import "../controls"
import "."
import "../../resources/font"

Item {
    id: l3DItem
    width: 50
    height: 50

    property string    title: ""
    property string    imgUrl: ""

    property int       margin: 0
    property int       spacing: 0
    property color     textColor: "white"
    property int       fontSize: 10

    property color     hoverColor: "#353535"

    signal clicked()

    Image {
        id: iconImage
        x: margin
        y: margin
        z: 1
        width: parent.width - 2*margin
        height: parent.height - 2*margin
        fillMode: Image.PreserveAspectFit
        source: imgUrl
    }

    Text {
        id: text
        x: margin
        y: parent.height/2 - paintedHeight/2
        z: 1
        width: paintedWidth
        height:paintedHeight
        text: title
        font.family: "Tahoma"
        color: textColor
        font.pointSize: fontSize
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }

    MouseArea {
        id: mouseArea
        x: 0
        y: 0
        width: btnHoverRect.width
        height: btnHoverRect.height
        anchors.fill: parent
        hoverEnabled: parent.enabled
        visible: true

        onEntered: {
            btnHoverRect.visible = true
        }
        onExited: {
            btnHoverRect.visible = false
            btnCustomHoverRect.visible = false
        }

        onHoveredChanged: {
        }

        onPressed: {
            btnCustomHoverRect.visible = true
        }

        onClicked: {
            l3DItem.clicked()
        }
    }

    Rectangle {
        id: btnHoverRect
        anchors.fill: parent
        color: hoverColor
        visible: false
        //border.color: hoverBorderColor
        //border.width: 1
    }

    Rectangle {
        id: btnCustomHoverRect
        anchors.fill: parent
        color: "#787272"
        visible: false
    }    
}
