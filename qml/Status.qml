import QtQuick 2.5
import QtQuick.Controls 1.4
import "./components"

Item {
    width: 1280
    height: 65
    property int minWidth: 1080
    property int minHeight: 720
    property string elapseText: qsTr("00:00:00.000")
    property string fpsText: qsTr("0 fps")
    property int levelText: 0
    property bool isPlaying: false
    property bool isRecording: false
    property bool isStreaming: false
	property bool isOculusMode: false

    Rectangle {
        anchors.fill: parent
        color:"#1f1f1f"
    }

    Rectangle {
        id: exampleTimeLineImage
        anchors.centerIn: parent
        color:"#1f1f1f"
        width: 650
        height: 60

        Label{
            id:rateID
            //anchors.left: elapsedTextID.right
            anchors.verticalCenter: parent.verticalCenter
            //anchors.leftMargin: 150
            text: "Rate:"
            font.pixelSize: 25
            color: "steelblue"
        }
        Label{
            id:rateTextID
            anchors.left: rateID.right
            anchors.verticalCenter: parent.verticalCenter
            anchors.leftMargin: 4
			width: 140
            text: fpsText
            font.pixelSize: 35
            color: "steelblue"
        }


        L3DButton {
            id: playAndpauseItem
            y: (parent.height - height) / 2
            anchors.left:  rateTextID.right
            anchors.leftMargin: 50
            width: 70
            height: 70
            margin: 10
            z: 1

            //imgUrl: (isPlaying)?(isRecording?"../../resources/ico_grey_pause.png":"../../resources/pause.png"):("../../resources/start.png")
            imgUrl: (isPlaying)?("../../resources/pause.png"):("../../resources/start.png")

            onClicked: {

                if (isPlaying && isRecording)
                    isRecording = false

                isPlaying = !isPlaying;
                if(isPlaying){
                    setPlayMode();
                }else{
                    setPauseMode();
                }
            }
        }


        L3DButton {
            id: recordItem
            width: 70
            height: 70
            margin: 10
            y: (parent.height - height) / 2
            anchors.left: playAndpauseItem.right
            //imgUrl: isRecording? ("../../resources/ico_stop_record.png") : (isPlaying ? "../../resources/ico_start_record.png": "../../resources/ico_grey_record.png")
            imgUrl: isRecording? ("../../resources/ico_stop_record.png"):(isPlaying? "../../resources/ico_start_record.png": "../../resources/ico_grey_record.png")
            enabled: (isRecording && !isStreaming)? true : ((isPlaying && !isStreaming) ? true: false)

            onClicked: {
                if (!isRecording){
                    if (qmlMainWindow.startRecordTake()) {
                        isRecording = true;
                    }
                }
                else {
                    stopRecordTake(centralItem.getTakeManangementComment());
                }
            }
        }

        L3DButton {
            id: streamingItem
            width: 70
            height: 70
            margin: 10
            y: (parent.height - height) / 2
            anchors.left: recordItem.right
            imgUrl: isStreaming? ("../../resources/ic_streaming_stop.png"): (isPlaying? "../../resources/ic_streaming_start.png": "../../resources/ic_streaming_pause.png")
            enabled: (isStreaming && !isRecording)? true: ((isPlaying && !isRecording)? true: false)

            onClicked: {

                if (isStreaming) {
                    //stop streamnig
                    isStreaming = false
                    centralItem.closeStreamingbox()
                    qmlMainWindow.stopStreaming()

                } else {
                    //start streaming
                    centralItem.createStreamingbox()
                }
            }
        }

        Item {
            id: elapsedTimeItem
            width: 310
            height: parent.height
            anchors.left: streamingItem.right
            anchors.leftMargin: 10

            Label{
                id:elapsedID
                text: "Elapsed:"
                font.pixelSize: 25
                color: "#dddddd"
                anchors.verticalCenter: parent.verticalCenter
            }

            Label{
                id:elapsedTextID
                anchors.left: elapsedID.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.leftMargin: 4
                text: elapseText
                font.pixelSize: 35
                color: "white"
            }
        }
    }

    L3DButton {
        id :takeMgrItem
        y: (parent.height - height) / 2
        anchors.left: exampleTimeLineImage.right
        anchors.leftMargin: 150
        z: 1
        imgUrl: "../../resources/ico_take.png"

        width: 70
        height: 70
        margin: 10

        onClicked: {
            centralItem.createTakeManagementSettingbox();            
            updateTakeMgr();
       }
    }

    L3DButton {
        id: oculusItem
        y: (parent.height - height) / 2
        anchors.left: takeMgrItem.right
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

	 function getOculus()
    {
        if(qmlMainWindow.getOculus())
        {
            //oculusSwitch.checked = true;
        }
        else
        {
            //oculusSwitch.checked = false;
        }
    }

    function getOculusStatus()
    {
        if (qmlMainWindow.notifyMsg.split(":")[0] === "Error")
        {
            //oculusSwitch.checked = false;
            qmlMainWindow.enableOculus(false);
			isOculusMode = false;
        }

    }

    function setPauseMode(){
        qmlMainWindow.setPlayMode(1);
    }


    function setPlayMode(){
        qmlMainWindow.setPlayMode(2);
    }

    function setStartRecordTake(){
        var ret = qmlMainWindow.startRecordTake();
		if (ret) {
            isRecording = true;
			return true;
		} else {
			return false;
		}
    }

    function stopRecordTake(comment){
        isRecording = false;
        qmlMainWindow.stopRecordTake(comment);
    }

    function setPlayAndPause() {
        if(isPlaying){
            setPlayMode()
        } else {
            setPauseMode();
        }
    }

    function initPlayAndRecord(){
        isPlaying = false;
        isRecording = false;
    }

    function updateTakeMgr(){
        //qmlMainWindow.takManagement.loadTree(takeMgrModel,qmlMainWindow.getSessionRootPath());
    }

    function setStreamingItem(status) {
        isStreaming = status
    }
}
