import QtQuick 2.5
import QtQuick.Controls.Styles 1.4
import QtQuick.Controls 1.4
import QtQml.Models 2.2
import QtQuick.Dialogs 1.2
import "../controls"

ToolWindow{
    id: streaming
    width: 740
    height: 200
    z: 10
    windowState: qsTr("windowed")
    visible: false

    property string   title: "RTMP Streaming"
    property int      fontSize: 15
    property int      itemHeight: 30
    property int      spacing: 20
    property int      leftMargin: 20

    property bool     isCustom: false
    property bool     isYouTube: false

    property int        youTubeWidth: 3840
    property int        youTubeHeight: 2160
    property int        originPanoramaWidth: 4096
    property int        originPanoramaHeight: 2048

    property int        curWidth: -1

    property var        streamingOptionEnum : {
        {"RTMP": 1}
        {"WEBRTC": 2}
    }

    Item {
        id: streamingItem
        width: parent.width
        height: 30
        anchors.top: parent.top
        anchors.topMargin: 20

        Text {
            id: streamingText
            y: 12
            color: "#ffffff"
            text: qsTr("Server address")
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 14

        }

        FlatText {
            id: streamingPath
            width: parent.width * 0.55
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: streamingText.right
            anchors.leftMargin: spacing
			text: !streamingOptionSwitch.checked? "udp://127.0.0.1:8051":"localhost"
            onTextChanged: {
                qmlMainWindow.setStreamingPath(streamingPath.text)
            }

            onAccepted: {
            }          
        }

        Item {
            id: streamingOption
            width: parent.width * 0.35
            height: 30
            anchors.left: streamingPath.right
            anchors.leftMargin: 15

            Text {
                id: rtmpText
                color: "white"
                width: parent.width * 0.05
                text: qsTr("RTMP")
                horizontalAlignment: Text.AlignRight
                anchors.left: parent.left
                anchors.leftMargin: spacing
                anchors.verticalCenter: parent.verticalCenter
                font.pixelSize: 13
            }

            Switch{
                id: streamingOptionSwitch
                anchors.left: rtmpText.right
                anchors.leftMargin: 10
                width: 50
                height: 30
                checked: qmlMainWindow.applicationSetting.streamingMode
                //enabled: isCustom
                onCheckedChanged: {
                }
            }

            Text {
                id: webRTCText
                color: "white"
                width: 50
                text: qsTr("WebRTC")
                horizontalAlignment: Text.AlignRight
                anchors.left: streamingOptionSwitch.right
                anchors.leftMargin: 10
                anchors.verticalCenter: parent.verticalCenter
                font.pixelSize: 13
            }
        }
    }

    Item {
        id: resolutionConfig
        width: parent.width
        height: 30
        anchors.top: streamingItem.bottom
        anchors.topMargin: 20

        Text {
            id: resolutionTypeText
            y: 12
            color: "#ffffff"
            text: qsTr("Resolution")
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 14
        }

        ComboBox {
            id: resolutionTypeCombo
            anchors.left: resolutionTypeText.right
            anchors.leftMargin: leftMargin
            anchors.verticalCenter: parent.verticalCenter
            width: parent.width * 0.2
            height: 30
            model: ListModel {
                id: resolutionTypeModel
            }

            onCurrentTextChanged:
            {

            }

            onCurrentIndexChanged: {
                setResolutionType(currentIndex)
            }
        }

        Text {
            id: panoWidthText
            y: 12
            color: "#ffffff"
            text: qsTr("Width")
            anchors.left: resolutionTypeCombo.right
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 14
        }

        FlatText {
            id: panoWidth
            width: parent.width * 0.1
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: panoWidthText.right
            anchors.leftMargin: 10
            enabled: isCustom

            onEditingFinished: {
                if (isYouTube)
                    panoHeight.text = text * (9/16)
                else
                    panoHeight.text = text * (1/2)

                curWidth = text;
            }

            onAccepted: {
                //console.log("New capture path: ", text)
                //qmlMainWindow.setSessionRootPath(text);
            }
        }

        Text {
            id: panoHeightText
            y: 12
            color: "#ffffff"
            text: qsTr("Height")
            anchors.left: panoWidth.right
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 14
        }

        FlatText {
            id: panoHeight
            width: parent.width * 0.1
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: panoHeightText.right
            anchors.leftMargin: 10
            enabled: isCustom


            onEditingFinished: {
                if (isYouTube)
                    panoHeight.text = text * (16/9)
                else
                    panoHeight.text = text * (2)
            }

            onTextChanged: {
                qmlMainWindow.setStreamingPath(streamingPath.text)
            }

            onAccepted: {
                //console.log("New capture path: ", text)
                //qmlMainWindow.setSessionRootPath(text);
            }


        }

        Text {
            id: wowzaText
            color: "white"
            width: parent.width * 0.05
            text: qsTr("Wowza")
            horizontalAlignment: Text.AlignRight
            anchors.left: panoHeight.right
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }

        Switch{
            id: serverTypeSwitch
            anchors.left: wowzaText.right
            anchors.leftMargin: 10
            width: 50
            height: 30
            checked: isYouTube
            enabled: isCustom
            onCheckedChanged: {
                if (checked) {
                    isYouTube = true
                    panoWidth.setText(youTubeWidth)
                    panoHeight.setText(youTubeHeight)
                }
                else {
                    isYouTube = false
                    panoWidth.setText(originPanoramaWidth)
                    panoHeight.setText(originPanoramaHeight)
                }
            }
        }

        Text {
            id: youTubeText
            color: "white"
            width: 50
            text: qsTr("YouTube")
            horizontalAlignment: Text.AlignRight
            anchors.left: serverTypeSwitch.right
            anchors.leftMargin: 10
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }
    }

    property string backColor: "#262626"

    Item {
        id: okItem
        width: 80
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 10
        height: 30
        anchors.topMargin: 10
        anchors.right: cancelItem.left
        anchors.rightMargin: spacing

        Rectangle {
            id: okHoverRect
            x: 0
            width: parent.width
            height: parent.height
            anchors.fill: parent
            color: "#373737"
            visible: false
            border.color: "#4e8d15"
            border.width: 1
            z: 1
        }


        Text {
            color: "#ffffff"
            text: "OK"
            z: 1
            font.pointSize: 10
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.fill: parent
        }



        Rectangle {
            id: okRect
            width: parent.width
            height: parent.height
           anchors.fill: parent
            color: "#373737"

            MouseArea {
                id: okMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true
                onHoveredChanged: {
                    isHovered = !isHovered
                    if(isHovered){
                        okHoverRect.visible = true;
                    }else{
                        okHoverRect.visible = false;
                    }
                }

                onClicked: {
                    root.isStartStreaming = false

                    statusItem.isPlaying = false
                    statusItem.setPauseMode()

                    maskID.visible = true
                    busyLabel.text = "Connecting to the RTMP server, please wait a moment..."

                    streamingTimer.restart()
                }
            }
        }

        Timer {
            id: streamingTimer
            interval: 100
            running: false
            repeat: false
            onTriggered: {

                var streamingMode
                if (streamingOptionSwitch.checked)
                    streamingMode = 2 // WebRTC
                else
                    streamingMode = 1 // RTMP

                if (qmlMainWindow.startStreaming(streamingPath.text, panoWidth.text, panoHeight.text, streamingMode)) {
                    statusItem.setStreamingItem(true) // connected on RTMP Server
                    streaming.visible = false

                    statusItem.isPlaying = true
                    statusItem.setPlayMode()
                }

                maskID.visible = false
            }
        }
    }

    Item {
        id: cancelItem
        width: 80
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 10
        height: 30
        anchors.topMargin: 10
        anchors.right: parent.right
        anchors.rightMargin: spacing

        Rectangle {
            id: cancelHoverRect
            x: 0
            width: parent.width
            height: parent.height
            anchors.fill: parent
            color: "#373737"
            visible: false
            border.color: "#4e8d15"
            border.width: 1
            z: 1
        }

        Text {
            color: "#ffffff"
            text: "Cancel"
            z: 1
            font.pointSize: 10
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.fill: parent
        }

        Rectangle {
            id: cancelRect
            width: parent.width
            height: parent.height
            anchors.fill: parent
            color: "#373737"

            MouseArea {
                id: cancelMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true
                onHoveredChanged: {
                    isHovered = !isHovered
                    if(isHovered){
                        cancelHoverRect.visible = true;
                    }else{
                        cancelHoverRect.visible = false;
                    }
                }

                onClicked: {
                    streaming.visible = false
                }
            }
        }
    }

    function getStreamingPath() {
        return streamingPath.text
    }

    function setStreamingPath(str) {
        streamingPath.setText(str)
    }

    function createResoltuionTypeModel(){
        resolutionTypeModel.clear();

        resolutionTypeModel.append({"text": "Stiching Resolution"})
        resolutionTypeModel.append({"text": "YouTube 2K"})
        resolutionTypeModel.append({"text": "YouTube 4K"})
        resolutionTypeModel.append({"text": "Custom"})
     }

    function setResolutionType(index) {
        switch (index) {
        case 0:
            panoWidth.setText(originPanoramaWidth)
            panoHeight.setText(originPanoramaHeight)
            isYouTube = false
            isCustom = false
            break
        case 1:
            panoWidth.setText(youTubeWidth/2)
            panoHeight.setText(youTubeHeight/2)
            isYouTube = true
            isCustom = false
            break
        case 2:
            panoWidth.setText(youTubeWidth)
            panoHeight.setText(youTubeHeight)
            isYouTube = true
            isCustom = false
            break
        case 3:
            isCustom = true
            isYouTube = false
            break
        }
    }

    function initializeResolution(width, height) {

        originPanoramaWidth = width
        originPanoramaHeight = height

        panoWidth.setText(originPanoramaWidth)
        panoHeight.setText(originPanoramaHeight)
    }
}
