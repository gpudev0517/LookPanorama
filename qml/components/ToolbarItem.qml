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
    id: toolbarItem
    width: imgUrl == ""? text.paintedWidth + 2* margin: iniHeight + text.paintedWidth + margin - spacing
    height: iniHeight

    property string    title: ""
    property string    imgUrl: ""
    property int       iniHeight: 48

    property string    tooltip: ""


    property  bool     isUsedImageButton: false
    property  int      customMargin: 0


    property string    theme: "Default"
    property bool      autoHide

    property color     hoverColor: "#353535"
    property color     hoverBorderColor : "#4e8d15"

    property int       fontSize: 10
    property color     textColor: "white"
    property string    fontURL: ""
    property string    systemFont: "MS Sans Serif"

    property int       margin: 16
    property int       spacing: 6
    property int       fixedMargin: 10

    property int       textWidth
    property int       imgWidth

    signal clicked()

    onThemeChanged: {
        setTextColor()
    }   

    FontLoader {
        id: localFont
        source: fontURL
    }

    FontLoader {
        id: fixedFont
        name: systemFont
    }

    Image {
        id: iconImage
        x: margin
        y: margin

        z: 1
        width: iniHeight - 2*margin
        height: iniHeight - 2*margin
        visible: true
        fillMode: Image.PreserveAspectFit
        source: imgUrl
    }

    Text {
        id: text
        x: (imgUrl == "")?margin:iconImage.x + iconImage.width + spacing
        y: iniHeight/2 - paintedHeight/2
        z: 1
        width: paintedWidth
        height:paintedHeight
        text: title
        font.family: fontURL == ""? "Tahoma": localFont.name
        color: textColor
        font.pointSize: fontSize
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter

        /*
        font.capitalization: {
            if (theme == "Default")
                return Font.Capitalize
            else if (theme == "VRSE")
                return Font.AllUppercase
        }
        */

        Component.onCompleted: {
            textWidth = paintedWidth
        }
    }

    MouseArea {
        id: mouseArea
        x: 0
        y:0
        width: btnHoverRect.width
        height: btnHoverRect.height
        anchors.fill: parent
        hoverEnabled: true
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
            toolbarItem.clicked()
        }
    }

    Image {
        id: iniImage
        visible: false
        source: imgUrl
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

    function setAttributes() {

        width = iniHeight + text.paintedWidth + margin - spacing
        height = iniHeight


        if (imgUrl == "")
            width = text.paintedWidth + 20

        if (isUsedImageButton)
            width = iniHeight
    }

    function setTextColor() {
        if (theme == "Default")
            textColor = "white"
        else if (theme == "VRSE")
            textColor = "#d0e3ef"
    }
}
