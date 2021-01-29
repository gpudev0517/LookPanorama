import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQml 2.2
import "./components"

Item {
    width: 1280
    height: 65
    visible: false
    property int minWidth: 1080
    property int minHeight: 720
    property string elapseText: qsTr("00:00:00.000")
    property string fpsText: qsTr("0 fps")
    property string lengthText: qsTr("00:00:00.000")
    property int levelText: 0
    property int sliderValue: 0
    property bool isPlaying: false
    property bool isOculusMode: false

    Rectangle {
        anchors.fill: parent
        color:"#1f1f1f"
    }

    Rectangle {
        id: playTimeLineImage
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top:parent.top
        color:"#1f1f1f"
        width: 950
        height: 70

        Label {
            id:rateID
            //anchors.left: elapsedTextID.right
            anchors.verticalCenter: parent.verticalCenter
            //anchors.leftMargin: 150
            text: "Rate:"
            font.pixelSize: 25
            color: "steelblue"
        }
        Label {
            id:rateTextID
            anchors.left: rateID.right
            anchors.verticalCenter: parent.verticalCenter
            anchors.leftMargin: 4
            width: 140
            text: fpsText
            font.pixelSize: 35
            color: "steelblue"
        }

        Item {
            id: prevItem
            width: 50
            height: 50
            y: (parent.height - height) / 2
            anchors.left: rateTextID.right
            anchors.leftMargin: 50

            Image {
                id:  prevImage
                width: 40
                height: 40
                anchors.centerIn: parent
                visible: true
                z: 1
                source: "../resources/media-skip-backward.png";
                fillMode: Image.PreserveAspectFit

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        prev();
                    }
                }
            }
        }

        Item {
            id: backwardItem
            width: 50
            height: 50
            y: (parent.height - height) / 2
            anchors.left: prevItem.right

            Image {
                id:  backwardImage
                width: 40
                height: 40
                anchors.centerIn: parent
                visible: true
                z: 1
                source: "../resources/media-seek-backward.png";
                fillMode: Image.PreserveAspectFit

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        backward();
                    }
                }
            }
        }

        Item {
            id: playAndpauseItem
            width: 50
            height: 50
            y: (parent.height - height) / 2
            anchors.left: backwardItem.right

            Image {
                id:  playImage
                width: 40
                height: 40
                anchors.centerIn: parent
                visible: true
                z: 1
                source: (isPlaying)?("../resources/media-playback-pause.png"):("../resources/media-playback-start.png");
                fillMode: Image.PreserveAspectFit

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        if (isPlaying) {
                            pause();
                        } else {
                            play();
                        }
                        isPlaying = !isPlaying;

                        seekMonitorTimer.restart();
                    }
                }
            }
        }

        Item {
            id: stopItem
            width: 50
            height: 50
            y: (parent.height - height) / 2
            anchors.left: playAndpauseItem.right

            Image {
                id: stopImage
                width: 40
                height: 40
                anchors.centerIn: parent
                visible: true
                z: 1
                source: "../resources/media-playback-stop.png";
                fillMode: Image.PreserveAspectFit

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        isPlaying = false;
                        stop();
                    }
                }
            }
        }

        Item {
            id: forwardItem
            width: 50
            height: 50
            y: (parent.height - height) / 2
            anchors.left: stopItem.right

            Image {
                id:  forwardImage
                width: 40
                height: 40
                anchors.centerIn: parent
                visible: true
                z: 1
                source: "../resources/media-seek-forward.png";
                fillMode: Image.PreserveAspectFit

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        forward();
                    }
                }
            }
        }

        Item {
            id: nextItem
            width: 50
            height: 50
            y: (parent.height - height) / 2
            anchors.left: forwardItem.right

            Image {
                id:  nextImage
                width: 40
                height: 40
                anchors.centerIn: parent
                visible: true
                z: 1
                source: "../resources/media-skip-forward.png";
                fillMode: Image.PreserveAspectFit

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        next();
                    }
                }
            }
        }

        Item {
            id: elapsedTimeItem
            width: 310
            height: parent.height
            anchors.left: nextItem.right
            anchors.leftMargin: 110

            Label {
                id:elapsedID
                text: "Elapsed:"
                font.pixelSize: 25
                color: "#dddddd"
                anchors.verticalCenter: parent.verticalCenter
            }
            Label {
                id:elapsedTextID
                anchors.left: elapsedID.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.leftMargin: 4
                text: elapseText
                font.pixelSize: 35
                color: "white"
            }
        }

        L3DButton {
            id: oculusItem
            y: (parent.height - height) / 2
            anchors.left: elapsedTimeItem.right
            anchors.leftMargin: 50
            imgUrl: "../../resources/icon_oculus.png"

            width: 70
            height: 70
            margin: 10

            onClicked: {
                root.onOculusWait();
                oculusTimer.restart();

                isOculusMode = !isOculusMode;
                qmlMainWindow.enableOculus(isOculusMode);
                if(isOculusMode){
                   getOculusStatus();
                }
           }
        }
    }

    Item {
        id:lengthItem
        width: 150
        height: parent.height/2
        anchors.top:playTimeLineImage.top
        anchors.topMargin: 50
        anchors.right: parent.right
        Label{
            id:lengthTextID
            anchors.right: parent.right
            anchors.rightMargin:100
            text: lengthText
            font.pixelSize: 15
            color: "white"
        }
        Label{
            id:lengthID
            anchors.right: lengthTextID.left
            text: "Length:"
            font.pixelSize: 15
            color: "white"
        }
    }

    Slider {
        id:seekSlider
        width:parent.width-100
        height:20
        anchors.right: lengthItem.right
        anchors.left: parent.left
        anchors.top: playTimeLineImage.bottom
        anchors.leftMargin:100
        anchors.rightMargin:50
        value: sliderValue
        onValueChanged:
        {
			if (pressed) {
				seek(value);
			}
        }
    }
    Rectangle {
        id: spliter
        x: 50
        width: parent.width - 50
        height: 2
        color: "#2b2b2b"
    }

    MouseArea {
        id: resizeMouseArea
        width: 5
        height: 5
        y: parent.height - height
        x: parent.width - width

        hoverEnabled: true
        onHoveredChanged: {
            if(!root.isFullScreen)
                cursorShape = Qt.SizeFDiagCursor
        }

        property bool isMoveMode: false
        property var clickPos: "1,1"
        property real movedX: 0
        property real movedY: 0

        onPressed: {
            if(mouse.button == Qt.LeftButton && !root.isFullScreen) {
                isMoveMode = true
                clickPos  = mapToItem(statusItem.parent, mouse.x,mouse.y)
            }
        }

        onPositionChanged: {
            if(isMoveMode) {
                var x = mouse.x
                var y = mouse.y
                var parentPos = mapToItem(statusItem.parent, x, y)
                var delta = Qt.point(parentPos.x-clickPos.x - movedX, parentPos.y-clickPos.y - movedY)

                if (root.width + delta.x >= minWidth) {
                    movedX += delta.x
                    root.width += delta.x
                }
                if (root.height + delta.y >= minHeight) {
                    movedY += delta.y
                    root.height += delta.y
                }
            }
        }

        onReleased: {
            if(mouse.button == Qt.LeftButton) {
                isMoveMode = false
                movedX = 0
                movedY = 0
            }
        }
    }

    Rectangle {
        id: resizeLine
        width: 30
        height: 2
        x: parent.width - width + 4
        y: parent.height - height - 3
        color: "#1f1f1f"
        rotation: -45
    }

    Rectangle {
        id: resizeLine2
        width: 20
        height: 2
        x: parent.width - width + 4
        y: parent.height - height - 3
        color: "#1f1f1f"
        rotation: -45
    }

    Timer {
        id: seekMonitorTimer
        interval: 1000
        running: true
        repeat: true
    }

    function resetPlayback() {
        isPlaying = false;
        lengthText = getDurationString();

        seekSlider.minimumValue = 0;
        seekSlider.maximumValue = getDuration();
    }


    function pause()
    {
		qmlMainWindow.setPlaybackMode(1);
    }

    function play()
    {
		qmlMainWindow.setPlaybackMode(2);
    }

    function stop()
    {
		qmlMainWindow.stopPlayback();
    }

    function prev()
    {
		qmlMainWindow.setPlaybackPrev();
		resetPlayback();
    }

    function next()
    {
		qmlMainWindow.setPlaybackNext();
		resetPlayback();
    }

    function backward()
    {
		qmlMainWindow.setPlaybackBackward();
		isPlaying = false;
    }

    function forward()
    {
		qmlMainWindow.setPlaybackForward();
		isPlaying = false;
    }

    function seek(value)
    {
		qmlMainWindow.setSeekValue(value);
    }

    function getDuration()
    {
		return qmlMainWindow.getDuration();
    }

    function getDurationString()
    {
		return qmlMainWindow.getDurationString();
    }
}
