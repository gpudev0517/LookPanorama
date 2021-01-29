import QtQuick 2.0

//Canvas {
//    id: canvas
//    width: parent.width
//    height: parent.height
//    z: 2
//    property var prePosX
//    property var prePosY
//    property var curPosX
//    property var curPosY
//    onPaint: {
//        var ctx = getContext("2d")

//        // setup the stroke
//        ctx.strokeStyle = "#4e8d15"

//        // create a path
//        ctx.beginPath();
//        ctx.moveTo(prePosX,prePosY);
//        ctx.lineTo(curPosX,curPosY);
//        //ctx.reset();
//        // stroke path
//        ctx.stroke()
//     }

//}

Item {
    id: bannerPoint
    property int xPos: 0
    property int yPos: 0
    property int lineSize: 10
    Rectangle {
        x: xPos - (lineSize - 1) / 2
        y: yPos
        width: 10
        height: 1
		color: "#00ff00"
    }

    Rectangle {
        x: xPos
        y: yPos - (lineSize - 1) / 2
        width: 1
        height: 10
        color: "#00ff00"
    }
}
