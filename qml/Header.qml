import QtQuick 2.5
import QtQuick.Window 2.2

Item {
    width: 1280
    height: 32
    z: 1
    property bool isCamFull: false

    property bool isVisibledApplicationSetting: false

    onWidthChanged: {
        sphericalView.updateWeightmapSettingsbox();
        centralItem.updateTakeManagementSettingbox();
        onClearHovered();
    }

    Rectangle {
        anchors.fill: parent
        color:"#000000"
    }

    Rectangle {
        id: headerBottom1
        y: 31
        width: parent.width
        height: 1
        color: "#0c0c0c"
    }

    Image {
        id: winGroupCtrlImage
        anchors.right: parent.right
        width: 144
        height: 31
        z: 1
        fillMode: Image.PreserveAspectFit
        source: "../resources/btn_group_wincontrols.png"

        MouseArea {
            id: minMouseArea
            x: 0
            y: 1
            width: 48
            height: 31
            hoverEnabled: true
            onHoveredChanged:  {
                onMinHovered()
            }
            onClicked: onMinimize()
        }

        MouseArea {
            id: maxMouseArea
            x: 48
            y: 1
            width: 48
            height: 31
            hoverEnabled: !isCamFull
            enabled: !isCamFull
            onHoveredChanged: {
                onMaxHovered()
            }
            onClicked: {
                onMaximize();
            }
        }

        MouseArea {
            id: closeMouseArea
            x: 96
            y: 1
            width: 48
            height: 31
            hoverEnabled: true
            onClicked: close()
            onHoveredChanged: {
                onCloseHovered()
            }
        }

        Image {
            id: restoreImage
            x: 65
            y: 11
            width: 14
            height: 13
            visible: true
            fillMode: Image.PreserveAspectFit
            source: "../resources/btn_restore.png"
        }
    }

    Image {
        id: applicationSettingImage
        anchors.right: winGroupCtrlImage.left
        anchors.rightMargin: 10
        x: 30
        y: 7
        z: 1
        width: 30
        height: 15
        visible: true
        fillMode: Image.PreserveAspectFit
        source: "../resources/applicationSetting.png"

        MouseArea {
            id: applicationSettingMouseArea
            x: 0
            y: 1
            width: 30
            height: 15
            //anchors.fill: parent
            hoverEnabled: true
            onClicked: {
                onApplicationSetting()
            }
            onHoveredChanged: {
                onApplicationSettingHovered()
            }
        }
    }

    /*Image {
        id: titleText
        x: 65
        y: 0
        fillMode: Image.PreserveAspectFit
        source: "../resources/icon_title.png"
    }*/
    /*Text{
        id: titleText
        x: 65
        y: 0
        width: 80
        height:  parent.height
        text: qsTr(headerTitle)
        font.pointSize: 17
        verticalAlignment: Text.AlignVCenter
        color: "#ffffff"
    }*/

    Rectangle {
        id: tipMessageRectangle
        x: 1198
        y: -63
        width: 82
        height: 32
        color: "#000000"
        radius: 1

        Text {
            id: tipText
            x: 4
            y: 5
            color: "#ffffff"
            text: qsTr("Minimize")
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.pixelSize: 19
        }
    }

    MouseArea {
        anchors.rightMargin: 144
        anchors.fill: parent
        enabled: !isCamFull

//      drag.target: parent
//      drag.axis: Drag.XAndYAxis

        property bool isMoveMode: false
        property var  curPos: "0,0"
        property var  deltaPos: "0,0"

        onPressed: {
            if(mouse.button == Qt.LeftButton && !root.isFullScreen) {
                isMoveMode = true
                curPos=Qt.point(root.x,root.y)
                deltaPos = Qt.point(mouse.x,mouse.y)
            }
        }

        onPositionChanged: {
            if(!isMoveMode) return
               curPos=Qt.point(curPos.x + mouse.x-deltaPos.x,curPos.y + mouse.y - deltaPos.y)
            root.setX(curPos.x)
            root.setY(curPos.y)
        }

        onReleased: {
            if(mouse.button == Qt.LeftButton) {
                isMoveMode = false
            }
        }

        onDoubleClicked: onMaximize()

    }

    Rectangle {
        id: winGroupHoverRetangle
        x: 1136
        y: -31
        width: 48
        height: 31
        color: "#1f1f1f"
    }

    Rectangle {
        id: closeHoverRectangle
        y: 0
        anchors.right: parent.right
        anchors.rightMargin: 0
        width: 48
        height: 31
        color: "#c75050"
        visible: false
    }

    Rectangle {
        id: maximizeHoverRectangle
        y: 0
        anchors.right: closeHoverRectangle.left
        anchors.rightMargin: 0
        width: 48
        height: 31
        color: "#1f1f1f"
        visible: false
    }

    Rectangle {
        id: minimizeHoverRectangle
        y: 0
        anchors.right: maximizeHoverRectangle.left
        anchors.rightMargin: 0
        width: 48
        height: 31
        color: "#1f1f1f"
        visible: false
    }

    function onMinimize(){
        root.showMinimized()
    }

    function onMaximize(){
        if (root.visibility === Window.Maximized) {
            root.visibility = Window.AutomaticVisibility
            root.isFullScreen = false
            restoreImage.visible = false
        } else if (root.visibility === Window.Windowed) {
            root.visibility = Window.Maximized
            root.isFullScreen = true
            restoreImage.visible = true
        }
    }

    property bool isMinHovered: false
    property bool isMaxnHovered: false
    property bool isCloseHovered: false
    property int hoverType:0
    property bool isApplicationSettingHovered: false

    Timer {
        id: delaytimer
        interval: 1500
        running: false
        repeat: false
        onTriggered: {
            switch(hoverType) {
            case 1:
                setMinHover()
                break;
            case 2:
                setMaxHover()
                break;
            case 3:
                setCloseHover()
                break;
            case 4:
                setApplicationSettingHover()
                break;
            default:
                break;
            }
        }
    }

    function close() {
		qmlMainWindow.closeMainWindow();
        //root.close();		// This operation will be excute on onExitChanged event.
    }

    function onApplicationSetting() {

        applicationSettingWindow.visible = true
    }

    function onApplicationSettingHovered() {
        isApplicationSettingHovered = !isApplicationSettingHovered
        if(isApplicationSettingHovered) {

            winGroupHoverRetangle.x = applicationSettingImage.x - 2
            winGroupHoverRetangle.y = 5
            winGroupHoverRetangle.height = 20
            winGroupHoverRetangle.width = 35
            winGroupHoverRetangle.color = "#1f1f1f"

            hoverType = 4
            delaytimer.restart()

        }else{
            onClearHovered()
        }
    }

    function onMinHovered(){
        isMinHovered = !isMinHovered
        if(isMinHovered) {

            minimizeHoverRectangle.visible = true
            hoverType = 1
            delaytimer.restart()
        }else{
            onClearHovered()
        }
    }

    function setMinHover() {
        tipMessageRectangle.width = 82
        tipText.text = "Minimize"
        winGroupHoverRetangle.color = "#1f1f1f"
        tipMessageRectangle.y = minMouseArea.height + 2
        tipMessageRectangle.x= winGroupCtrlImage.x
    }

    function onMaxHovered() {
        isMaxnHovered = !isMaxnHovered
        if(isMaxnHovered) {

            maximizeHoverRectangle.visible = true
            hoverType = 2
            delaytimer.restart()
        }else{
            onClearHovered()
        }
    }
			
    function setMaxHover() {
        tipMessageRectangle.width = 85
        tipText.text = "Maximize"
        tipMessageRectangle.y = minMouseArea.height + 2
        tipMessageRectangle.x= winGroupCtrlImage.x + winGroupHoverRetangle.width
    }

    function onCloseHovered() {
        isCloseHovered = !isCloseHovered
        if(isCloseHovered) {


            closeHoverRectangle.visible = true
            hoverType = 3
            delaytimer.restart()
        }else{
            onClearHovered()
        }
    }

    function setCloseHover() {
        tipMessageRectangle.width = 55
        tipText.text = "Close"
        tipMessageRectangle.y = minMouseArea.height + 2
        tipMessageRectangle.x= winGroupCtrlImage.x + winGroupHoverRetangle.width*2 - 10
    }

    function setApplicationSettingHover() {
        tipMessageRectangle.width = 70
        tipText.text = "Setting"
        tipMessageRectangle.y = applicationSettingMouseArea.height + 2
        tipMessageRectangle.x = applicationSettingImage.x
    }

    function onClearHovered(){
		delaytimer.stop()
        winGroupHoverRetangle.y = -50
        tipMessageRectangle.y = -50
        minimizeHoverRectangle.visible = false
        maximizeHoverRectangle.visible = false
        closeHoverRectangle.visible = false
    }
}
