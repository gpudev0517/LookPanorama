import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4

ToolWindow{
    id: details
    width: 550
    height: 500
    z: 10
    windowState: qsTr("windowed")
    visible: false
    floatingWindowBorderColor: "#1f1f1f"

    property bool   isHover: false
    property int    camIndex1: 0
    property string pos

    property string title: "Details"
    property int    fontSize: 20
    property bool   isCreated: false
    property int    spacing: 15
    property var    ctxList: []
    property var    adPoints: []
    property int    adIndex: -1
    property var    clickPos
    property var    curPosX
    property var    curPosY
    property var    movedPos: "0,0"
    property real    alpha: 0.1
    property var    lineWidth: 30
    property var    lineStyle: Qt.rgba(0,0,0,alpha)

    TabView {
        id: tabView
        x: (parent.width - width) / 2
        y: 15
        width: parent.width - spacing * 2
        height: parent.height - spacing * 5
        onCurrentIndexChanged: {
            var str = qmlMainWindow.getWeightMapFile(currentIndex);
            //weightmapImage.source = "file:///F:/Projects/PanoOne/SVN/Source/Look3D/build12/"  + str;
        }

        style: TabViewStyle {
            frameOverlap: 1
            tab: Rectangle {
                color: styleData.selected ? "#4e8d15" :"#171717"
                border.color:  "#4e8d15"
                implicitWidth: 40
                implicitHeight: 30
                radius: 2
                Text {
                    id: text
                    anchors.centerIn: parent
                    text: styleData.title
                    color: styleData.selected ? "white" : "white"
                }
            }
            frame: Rectangle {
                color: "#FFFFFF"
                //color: "#000000"
                opacity: 1.0
            }
        }

        Canvas {
            id: back_image
            width: parent.width
            height: parent.height
            z: 2
            property string imgURL: "file:///E:/test.png"
            property var ctx: getContext("2d")

            Component.onCompleted: {
                loadImage(imgURL);
            }

            onImageLoaded: {
                ctx.drawImage(imgURL, 0, 0, parent.width, parent.height);
            }
        }

        Canvas {
            id: canvas
            width: parent.width
            height: parent.height
            z: 3
            property var prevPts: []
            property var curPts: []
            property var ctx: getContext("2d")
            property var beginPath: 0;

            Component.onCompleted: {
                ctx.lineWidth = lineWidth;
                ctx.strokeStyle = lineStyle;

                prevPts[0] = mapToItem(parent, -1, -1);
                prevPts[1] = mapToItem(parent, -1, -1);
                prevPts[2] = mapToItem(parent, -1, -1);
                curPts[0] = mapToItem(parent, -1, -1);
                curPts[1] = mapToItem(parent, -1, -1);
                curPts[2] = mapToItem(parent, -1, -1);
            }

            function updateStyle() {
                ctx.strokeStyle = lineStyle;
                ctx.lineWidth = lineWidth;
            }

            function updatePoints() {
                prevPts[0] = curPts[0];
                prevPts[1] = curPts[1];
                prevPts[2] = curPts[2];

                clearPoints();
            }

            function clearPoints() {
                curPts[0] = mapToItem(parent, -1, -1);
                curPts[1] = mapToItem(parent, -1, -1);
                curPts[2] = mapToItem(parent, -1, -1);
            }

            function saveToFile() {
                var toSave = toDataURL("image/png");
                //var result = save("file:///E:/result.png");
                var result = save(toSave);
                console.log("save result: "+ result + toSave);
            }

            onPaint: {
                if (prevPts[2].x === -1) return;

                if (beginPath === 0) {
                    ctx.beginPath();
                    ctx.moveTo(prevPts[2].x, prevPts[2].y);
                    beginPath++;
                } else if (curPts[2].x !== -1 && beginPath > 0) {
                    ctx.bezierCurveTo(curPts[0].x,curPts[0].y,curPts[1].x,curPts[1].y,curPts[2].x,curPts[2].y);
                    beginPath++;
                } else if (curPts[0].x === -1 && beginPath > 0){
                    ctx.stroke();
                    ctx.closePath();
                    clearPoints();
                }

                updatePoints();
            }

            MouseArea {
                anchors.fill: parent

                onPressed: {
                    movedPos  = mapToItem(parent,mouse.x,mouse.y);

                    if(pressedButtons != Qt.LeftButton) return;

                    canvas.prevPts[0] = mapToItem(parent, -1, -1);
                    canvas.prevPts[1] = mapToItem(parent, -1, -1);
                    canvas.prevPts[2] = movedPos;

                    canvas.beginPath = 0;
                }

                onReleased: {
                    canvas.clearPoints();
                    canvas.requestPaint();
                }

                onPositionChanged: {
                    movedPos  = mapToItem(parent,mouse.x,mouse.y);

                    if(pressedButtons != Qt.LeftButton) return;

                    if (canvas.curPts[0].x === -1) {
                        canvas.curPts[0] = movedPos;
                    } else if (canvas.curPts[1].x === -1) {
                        canvas.curPts[1] = movedPos;
                    } else {
                        canvas.curPts[2] = movedPos;
                        canvas.requestPaint();
                    }
                }
            }
        }
    }

    Item {
        id: alphaPlus
        y: 24
        anchors.right: alphaMin.left
        anchors.rightMargin: spacing / 2
        property int screenNum: 1
        property bool isSelected: false
        width: 28
        height: 20
        Rectangle {
            id: alphaPlusRectangle
            color: "#171717"
           // border.color:  "#4e8d15"
            implicitWidth: parent.width
            implicitHeight: parent.height

            Text {
                anchors.centerIn: parent
                text: "+"
                color: "white"
                font.pixelSize: 15
            }
        }

        MouseArea{
            anchors.fill: parent
            hoverEnabled: true
            onEntered: alphaPlusRectangle.border.color = "#4e8d15"
            onExited: alphaPlusRectangle.border.color = "#00000000"
            onClicked: {
                lineStyle = Qt.rgba(0,0,0,alpha);
                canvas.updateStyle();
            }
        }
    }

    Item {
        id: alphaMin
        anchors.right: parent.right
        anchors.rightMargin: spacing
        y: 24
        property int screenNum: 1
        property bool isSelected: false
        width: 28
        height: 20
        Rectangle {
            id: alphaMinRectangle
            color: "#171717"
            //border.color:  "#4e8d15"
            implicitWidth: parent.width
            implicitHeight: parent.height

            Text {
                anchors.centerIn: parent
                text: "-"
                color: "white"
                font.pixelSize: 15
            }
        }

        MouseArea{
            anchors.fill: parent
            hoverEnabled: true
            onEntered: alphaMinRectangle.border.color = "#4e8d15"
            onExited: alphaMinRectangle.border.color = "#00000000"
            onClicked: {
                lineStyle = Qt.rgba(255,255,255,alpha);
                canvas.updateStyle();
            }
        }
    }

    Item {
        id: resetItem
        width: 65
        anchors.top: tabView.bottom
        anchors.topMargin: spacing
        height: 30
        anchors.right: resetAllItem.left
        anchors.rightMargin: spacing / 2

        Rectangle {
            id: resetHoverRect
            x: 0
            width: parent.width
            height: parent.height
            anchors.fill: parent
            color: "#171717"
            visible: false
            border.color: "#4e8d15"
            border.width: 1
            z: 1
        }

        Text {
            id: resetText
            z: 1
            color: "#ffffff"
            text: "Reset"
            font.pointSize: 11
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.fill: parent
        }

        Rectangle {
            id: resetRect
            width: parent.width
            height: parent.height
           anchors.fill: parent
            color: "#171717"

            MouseArea {
                id: resetMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true
                onHoveredChanged: {
                    isHovered = !isHovered
                    if(isHovered){
                        resetHoverRect.visible = true;
                    }else{
                        resetHoverRect.visible = false;
                    }
                }

                onClicked: {
                    canvas.ctx.clearRect(0,0,canvas.width,canvas.height);
                    canvas.requestPaint();
                }
            }
        }
    }

    Item {
        id: resetAllItem
        width: 65
        anchors.top: tabView.bottom
        anchors.topMargin: spacing
        height: 30
        anchors.right: checkItem.left
        anchors.rightMargin: spacing / 2

        Rectangle {
            id: resetAllHoverRect
            x: 0
            width: parent.width
            height: parent.height
            anchors.fill: parent
            color: "#171717"
            visible: false
            border.color: "#4e8d15"
            border.width: 1
            z: 1
        }

        Text {
            id: resetAllText
            z: 1
            color: "#ffffff"
            text: "Reset All"
            font.pointSize: 11
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.fill: parent
        }

        Rectangle {
            id: resetAllRect
            width: parent.width
            height: parent.height
           anchors.fill: parent
            color: "#171717"

            MouseArea {
                id: resetAllMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true
                onHoveredChanged: {
                    isHovered = !isHovered
                    if(isHovered){
                        resetAllHoverRect.visible = true;
                    }else{
                        resetAllHoverRect.visible = false;
                    }
                }

                onClicked: {
                    canvas.saveToFile();
                }
            }
        }
    }


    Item {
        id: checkItem
        width: 65
        anchors.top: tabView.bottom
        anchors.topMargin: spacing
        height: 30
        anchors.right: cancelItem.left
        anchors.rightMargin: spacing / 2

        Rectangle {
            id: checkHoverRect
            x: 0
            width: parent.width
            height: parent.height
            anchors.fill: parent
            color: "#171717"
            visible: false
            border.color: "#4e8d15"
            border.width: 1
            z: -1
        }

        Rectangle {
            id: checkRect
            width: parent.width
            height: parent.height
           anchors.fill: parent
            color: "#171717"

            MouseArea {
                id: checkMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true
                onHoveredChanged: {
                    isHovered = !isHovered
                    if(isHovered){
                        checkRect.border.color = "#4e8d15";
                    }else{
                        checkRect.border.color = "#00000000";
                    }
                }

                onClicked: {

                }
            }

            Image {
                x: (parent.width - width) / 2
                y: (parent.height - height) / 2
                width: 25
                height: 25
                fillMode: Image.PreserveAspectFit
                source: "../../resources/check.png"
            }
        }
    }


    Item {
        id: cancelItem
        width: 65
        anchors.top: tabView.bottom
        anchors.topMargin: spacing
        height: 30
        anchors.right: parent.right
        anchors.rightMargin: spacing

        Rectangle {
            id: cancelHoverRect
            x: 0
            width: parent.width
            height: parent.height
            anchors.fill: parent
            color: "#171717"
            visible: false
            border.color: "#4e8d15"
            border.width: 1
            z: -1
        }

        Rectangle {
            id: cancelRect
            width: parent.width
            height: parent.height
           anchors.fill: parent
            color: "#171717"

            MouseArea {
                id: cancelMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true
                onHoveredChanged: {
                    isHovered = !isHovered
                    if(isHovered){
                        cancelRect.border.color = "#4e8d15"
                    }else{
                        cancelRect.border.color = "#00000000"
                    }
                }

                Image {
                    id: cancelImage
                    x: (parent.width - width) / 2
                    y: (parent.height - height) / 2
                    width: 25
                    height: 25
                    fillMode: Image.PreserveAspectFit
                    source: "../../resources/uncheck.png"
                }

                onClicked: {

                }
            }
        }
    }

    function createTab(){
        if(isCreated) return;
        isCreated = true;
        var cameraCnt = qmlMainWindow.getCameraCount();
        for(var i = 0; i < cameraCnt; i++){
            tabView.addTab(i + 1);
        }

     }

    function clearTab()
    {
        var cameraCnt = qmlMainWindow.getCameraCount();
        for(var i = cameraCnt; i > 0; i--){
            tabView.removeTab(i - 1);
        }
    }
}
