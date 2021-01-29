import QtQuick 2.0

Item {
    property var    movedPos: "0,0"

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
            var result = save("result.png");
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

    function clearWeightmap(){
        canvas.ctx.clearRect(0,0,canvas.width,canvas.height);
        canvas.requestPaint();
    }

    function saveImage(){
        canvas.saveToFile();
    }

}

