import QtQuick 2.5

Item{
	id: liveAudioItem
    width: root.width
    height: 30

    property string     title: "title"
    property string     titleTextColor
    property string     mixTxt
    property string     leftTxt
    property string     rightTxt
    property string     noneTxt
    property bool       isHovered: false
    property bool       isSelected: false
    property bool       isVolumeSelected: false
    property bool       isMixSelected: false
    property bool       isAudioLeft: true
    property bool       isAudioRight: false
    property int        clickCnt: 0
    property bool       checkSelect: false
    property var        selectAudio
    property bool       checkMix: false
    property bool       checkLeft: false
    property bool       checkRight: false
    property bool       checkAudio: false
    property int        liveAudioType: 1
    property bool       isClicked: checkSelect
    property int        spacing: 50
    property string     audioLeftColor
    property string     audioRightColor

    Rectangle {
        id: hoveredRectangle
        x: 20
        y: -5
        width: root.width - 50
        height: parent.height + 10
        color: "#1f1f1f"
        visible: false

    }
    Rectangle {
        id: selectRectangle
        x: 20
        y: -5
        width: root.width - 50
        height: parent.height + 10
        color: "#0e3e64"
        visible: checkSelect
    }
    Item {
        id: volumeItem
        x: 50
        y: (parent.height - height) / 2
        z: 5
        width: 30
        height: 30

        Image {
            id: muteImage
            width: 25
            height: 25
            source: "../../resources/mute.png"
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 1
            visible: false
            fillMode: Image.PreserveAspectFit
        }

        Image {
            id: volumeImage
            width: 25
            height: 25
            source: "../../resources/volume.png"
            x:(parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 1
            visible: true
            fillMode: Image.PreserveAspectFit
        }

        MouseArea {
            id: volumeArea
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

				liveAudioItem.toggleAudioSlotSelection();
            }
        }
    }

    Text {
        id: titleText
        x: volumeItem.x + volumeItem.width + 20
        y: (parent.height - height) / 2
        z: 1
        color: titleTextColor
        text: qsTr(title)
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        font.pixelSize: 15
    }

    Item{
        id: leftItem
        anchors.right: rightItem.left
        anchors.rightMargin: 50
        z: 5
        width: 40
        height: 30
        visible: checkSelect

        Text{
            id: leftText
            text: "Left"
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 1
            color: audioLeftColor
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
            font.pixelSize: 15

        }
        MouseArea {
            id: leftMouseArea
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
                isAudioLeft = !isAudioLeft
                if (isAudioLeft)
                {
                    if(isAudioRight)
                    {
                       leftText.color = "#ffffff"
                       liveAudioType = 0;
                        selectAudio = audioList.get(index);
                        selectAudio.audioType = 0;
                    }
                    else
                    {
                        leftText.color = "#ffffff"
                        liveAudioType = 1;
                        selectAudio = audioList.get(index);
                        selectAudio.audioType = 1;
                    }
                }
                else
                {
                    if(isAudioRight)
                    {
                        leftText.color = "#8a8a8a"
                        liveAudioType = 2;
                        selectAudio = audioList.get(index);
                        selectAudio.audioType = 2;
                    }
                    else
                    {
                        liveAudioType = 3;
                        selectAudio = audioList.get(index);
                        selectAudio.audioType = 3;
                        leftText.color = "#8a8a8a"
                    }
                }
            }

        }
    }
    Item{
        id: rightItem
        anchors.right: parent.right
        anchors.rightMargin: 100 + spacing / 2.5
        z: 5
        width: 40
        height: 30
        visible: checkSelect

        Text{
            id: rightText
            text: "Right"
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 1
            color: audioRightColor
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
            font.pixelSize: 15

        }
        MouseArea {
            id: rightMouseArea
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
                isAudioRight = !isAudioRight
                if (isAudioRight)
                {
                    if(isAudioLeft)
                    {
                        rightText.color = "#ffffff"
                        liveAudioType = 0;
                        selectAudio = audioList.get(index);
                        selectAudio.audioType = 0;
                    }
                    else
                    {
                        rightText.color = "#ffffff"
                        selectAudio = audioList.get(index);
                        selectAudio.audioType = 2;
                        liveAudioType = 2;
                    }
                }
                else
                {
                    if(isAudioLeft)
                    {
                        rightText.color = "#8a8a8a"
                        selectAudio = audioList.get(index);
                        selectAudio.audioType = 1;
                        //liveAudioType = 1;
                    }
                    else
                    {
                        selectAudio = audioList.get(index);
                        selectAudio.audioType = 3;
                        //liveAudioType = 3;
                        rightText.color = "#8a8a8a"
                    }
                }
            }

        }
    }

    MouseArea {
        id: mouseArea
        z: 2
        width: parent.width * 0.6
        height: parent.height
        hoverEnabled: true
        onHoveredChanged: {
            isHovered = !isHovered
            if(isHovered) {
                hoveredRectangle.visible = true
            }
            else {
                hoveredRectangle.visible = false
            }
        }
        onClicked:{
            liveAudioItem.toggleAudioSlotSelection();
        }

    }
	function toggleAudioSlotSelection() {
		isClicked = !isClicked;
        if(isClicked){
            titleText.color = "#ffffff"
            selectAudio = audioList.get(index);
            selectAudio.isSelect = true;
            selectRectangle.visible = true;
            leftItem.visible = true;
            selectAudio.audioType = 1;
            isAudioLeft = true;
            isAudioRight = false;
            selectAudio.selectAudioLeft = true;
            selectAudio.selectAudioRight = false;
            selectAudio.leftColor = "#ffffff";
            selectAudio.rightColor = "#8a8a8a"
            leftText.color = "#ffffff";
            rightText.color = "#8a8a8a";
            rightItem.visible = true;
        }else{
            selectAudio = audioList.get(index);
            selectAudio.isSelect = false;
            selectAudio.audioType = 3;
            selectAudio.selectAudioLeft = false;
            isAudioLeft = false;
            isAudioRight = false;
            selectAudio.selectAudioRight = false;
            selectAudio.leftColor = "#8a8a8a";
            selectAudio.rightColor = "#8a8a8a"
            leftText.color = "#8a8a8a";
            rightText.color = "#8a8a8a";
            clearAudioSettings();
        }
	}
    function clearAudioSettings(){
        titleText.color = "#8a8a8a";
        selectRectangle.visible = false;
        leftItem.visible = false;
        rightItem.visible = false;
    }
}
