import QtQuick 2.5

Item {
    width: 150
    height: 180

    property string     title: "title"
    property string     iconPath: "path"
    property int        index: 0
    property string     iniPath: ""
    property bool       isSelected: false

    Rectangle {
        id: hoveredRectangle
        z: 0
        width: parent.width
        height: parent.height
        color: "#0e3e64"
        visible: isSelected
    }

    Rectangle {
        id: templateRectangle
        x: (parent.width - width) / 2
        y: (parent.height - height) / 4
        z: 1
        width: parent.width - 20
        height: parent.height - 50
        color: "#171717"
        border.color: "#AAAAAA"
        border.width: 1

        Image {
            id: canvasImage
            width: 60
            height: 60
            opacity: 0.8
            scale: 1
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            x: 10
            y: 10
            z: 2
            fillMode: Image.PreserveAspectFit
            source: iconPath
        }
    }

    Text {
        id: templateText
        x: (parent.width - width) / 2
        y: templateRectangle.y + templateRectangle.height + 5
        z: 1
        color: "#ffffff"
        text: qsTr(title)
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        font.pixelSize: 16
    }

    MouseArea {
        id: mouseArea
        z: 2
        width: parent.width
        height: parent.height
        hoverEnabled: true        
        onClicked: onSetting()
    }

    function onSetting(){
        isSelected = !isSelected;               
        selectFavorite_Template(index, iniPath, isSelected);
    }

    function clearSelected(){
        forwardItem.visible = false;
        dshowSettingbox.state = "collapsed"
        videoSettingbox.state = "collapsed"
        imageSettingbox.state = "collapsed"
    }
}
