import QtQuick 2.5

Item{
    id: item1
    width: root.width
    height: 30

    property string title: "titleTxt"
    property bool isHovered: false
    property bool isSelected: false
    property bool isVolumeSelected: false
    property bool isMixSelected: false
    property bool isLeftSelected: false
    property bool isRightSelected: false
    property int clickCnt: 0

    Rectangle {
        id: hoveredRectangle
        x: 20
        width: root.width - 50
        height: parent.height
        color: "#1f1f1f"
        visible: false

    }
    Rectangle {
        id: selectRectangle
        x: 20
        width: root.width - 50
        height: parent.height
        color: "#0e3e64"
        visible: false
    }
    Item {
        id: videoItem
        x: 20
        y: (parent.height - height) / 2
        z: 5
        width: 30
        height: 30

        Image {
            id: videoImage
            width: 30
            height: 30
            source: "../../resources/icon_video_small.png"
            x:(parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 1
            fillMode: Image.PreserveAspectFit
        }

        MouseArea {
            id: videoArea
            anchors.fill: parent
            z: 10
            hoverEnabled: true
            onHoveredChanged: {
                cursorShape = Qt.PointingHandCursor
                isHovered = !isHovered
                if(isHovered) {
                    hoveredRectangle.visible = true
                }
                else {
                    hoveredRectangle.visible = false
                }

            }

            onClicked: {
                isVolumeSelected = !isVolumeSelected
                if(isVolumeSelected){
                    volumeImage.visible = true
                    muteImage.visible = false
                }else{
                    volumeImage.visible = false
                    muteImage.visible = true
                }
            }
        }
    }

    Text {
        id: titleText
        x: videoItem.x + videoItem.width + 20
        y: (parent.height - height) / 2
        z: 1
        color: "#8a8a8a"
        text: qsTr(title)
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        font.pixelSize: 15
    }

    MouseArea {
        id: mouseArea
        z: 2
        width: parent.width
        height: parent.height
        hoverEnabled: true
        onHoveredChanged: {
            cursorShape = Qt.PointingHandCursor
            isHovered = !isHovered
            if(isHovered) {
                hoveredRectangle.visible = true
            }
            else {
                hoveredRectangle.visible = false
            }
        }
        onClicked:{
            //toolbox.clearSelected()
            //backItem.visible = true
            //recent.state = "collapsed"
            clickCnt=clickCnt + 1
            if(clickCnt % 2){
              titleText.color = "#ffffff"
              selectRectangle.visible = true
            }else{
                titleText.color = "#8a8a8a"
                selectRectangle.visible = false
            }

        }
    }
}
