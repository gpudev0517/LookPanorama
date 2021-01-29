import QtQuick 2.5
import QtQuick.Layouts 1.1
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2

import "."
import "../components"
import "../components/QMLChartJs/QMLChartData.js" as ChartsData
import "../components/QMLChartJs/QChartJsTypes.js" as ChartTypes
import "../components/QMLChartJs"
import "../controls"



ToolWindow {
	id: lutControl
    width: 600
    height: 630
    z: 10
	windowState: qsTr("windowed")
    visible: true

	property string   title: "LUT Control"
	property int      fontSize: 15
	property int      itemHeight: 30
    property int      spacing: 20
    property color    textColor: "#ffffff"
    property var      currentCursorShape: Qt.CrossCursor
    property int      lblWidth: 60

	property int        leftMargin: 20
	property int        rightMargin: 0	
	property color      spliterColor: "#555555"
	property int        lblFont: 14		
	property var        opacityVal: 0.9	
    property int        spaceSize: 5
    property int        btnSize: 70

    property string     lutType: "Gray"
    property var        grayLutData: []
    property var        redLutData: []
    property var        greenLutData: []
    property var        blueLutData: []
    property var        initLutData: [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    property color      selectedBtnColor: "green"

    Rectangle {
        id: rectangle
        anchors.fill: parent
        color: "black"



        Rectangle {
            id: rectangleButtonGroup
            x: 0
            y: 0
            width: parent.width
            height: 40
            Layout.fillHeight: false
            Layout.fillWidth: false
            color: "#171717"
            opacity: 0.9

            Item {
                id: grayItem
                width: btnSize
                height: 30
                x: spaceSize
                y: 5

                //Layout.fillWidth: true
                //Layout.fillHeight: true

                Rectangle {
                    id: grayHoverRect
                    x: 0
                    width: parent.width
                    height: parent.height
                    anchors.fill: parent
                    color: selectedBtnColor
                    visible: false
                    border.color: "#4e8d15"
                    border.width: 1
                    z: 1
                }


                Text {
                    id: grayText
                    color: "#ffffff"
                    text: "Gray"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent



                }

                Rectangle {
                    id: grayRect
                    width: parent.width
                    height: parent.height
                    anchors.fill: parent
                    color: selectedBtnColor

                    MouseArea {
                        id: grayMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        visible: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                grayHoverRect.visible = true;
                            }else{
                                grayHoverRect.visible = false;
                            }
                        }

                        onClicked: {
	                        lutType = "Gray"

                            chart_line.requestPaint()
                            setSelectedBtnColor(lutType)

                            if (grayLutData.length == 0)
                                onReset(lutType);
                            else
                                updateLUT(grayLutData);
                        }
                    }
                }
            }

            Item {
                id: redItem
                Layout.fillWidth: true
                //Layout.fillHeight: true
                //anchors.fill: parent
                width: btnSize
                height: 30

                anchors.left: grayItem.right
                anchors.leftMargin: spaceSize
                y: 5

                Rectangle {
                    id: redHoverRect
                    x: 0
                    width: parent.width
                    height: parent.height
                    anchors.fill: parent
                    color: selectedBtnColor
                    visible: false
                    border.color: "#4e8d15"
                    border.width: 1
                    z: 1
                }


                Text {
                    id: redText
                    color: "#ffffff"
                    text: "Red"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent


                }

                Rectangle {
                    id: redRect
                    width: parent.width
                    height: parent.height
                    color: "#373737"

                    MouseArea {
                        id: redMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                redHoverRect.visible = true;
                            }else{
                                redHoverRect.visible = false;
                            }
                        }

                        visible: true
                        onClicked: {
                            lutType = "Red"

                            chart_line.requestPaint()
                            setSelectedBtnColor(lutType)

                            if (redLutData.length == 0)
                                onReset(lutType)
                            else
                                updateLUT(redLutData);
                        }
                    }
                }
            }

            Item {
                id: greenItem
                Layout.fillWidth: true
                //Layout.fillHeight: true
                //anchors.fill: parent
                width: btnSize
                height: 30

                anchors.left: redItem.right
                anchors.leftMargin: spaceSize
                y: 5

                Rectangle {
                    id: greenHoverRect
                    x: 0
                    width: parent.width
                    height: parent.height
                    anchors.fill: parent
                    color: selectedBtnColor
                    visible: false
                    border.color: "#4e8d15"
                    border.width: 1
                    z: 1
                }


                Text {
                    id: greenText
                    color: "#ffffff"
                    text: "Green"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent


                }

                Rectangle {
                    id: greenRect
                    width: parent.width
                    height: parent.height
                    color: "#373737"

                    MouseArea {
                        id: greenMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                greenHoverRect.visible = true;
                            }else{
                                greenHoverRect.visible = false;
                            }
                        }

                        visible: true
                        onClicked: {
                            lutType = "Green"
                            chart_line.requestPaint()
                            setSelectedBtnColor(lutType)

                            if (greenLutData.length == 0)
                                onReset(lutType);
                            else
                                updateLUT(greenLutData);
                        }
                    }
                }
            }

            Item {
                id: blueItem
                Layout.fillWidth: true
                //Layout.fillHeight: true
                width: btnSize
                height: 30

                anchors.left: greenItem.right
                anchors.leftMargin: spaceSize
                y: 5

                Rectangle {
                    id: blueHoverRect
                    x: 0
                    width: parent.width
                    height: parent.height
                    anchors.fill: parent
                    color: selectedBtnColor
                    visible: false
                    border.color: "#4e8d15"
                    border.width: 1
                    z: 1
                }


                Text {
                    id: blueText
                    color: "#ffffff"
                    text: "Blue"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent


                }

                Rectangle {
                    id: blueRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id:  blueMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        visible: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                blueHoverRect.visible = true;
                            }else{
                                blueHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            lutType = "Blue"
                            chart_line.requestPaint()
                            setSelectedBtnColor(lutType)

                            if (blueLutData.length == 0)
                                onReset(lutType);
                            else
                                updateLUT(blueLutData);
                        }
                    }
                }
            }

            Item {
                id: resetItem

                Layout.fillWidth: true
                //Layout.fillHeight: true
                width: btnSize
                height: 30

                anchors.left: blueItem.right
                anchors.leftMargin: spaceSize
                y: 5

                Rectangle {
                    id: resetHoverRect
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
                    id: resetText
                    color: "#ffffff"
                    text: "Reset"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent
                }

                Rectangle {
                    id: resetRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id:  resetMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        visible: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                resetHoverRect.visible = true;
                            }else{
                                resetHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            chart_line.requestPaint()
                            onReset(lutType);
                        }
                    }
                }
            }

            Item {
                id: spliter
                y: 5
                width: 5
                height: 30
                anchors.left: resetItem.right
                anchors.leftMargin:  3*spaceSize
                Rectangle{
                    color: "#1f1f1f"
                    x: 1
                    height: parent.height
                    width: 1

                }
                Rectangle{
                    color: "#3f3f3f"
                    x: 2
                    height: parent.height
                    width: 1
                }
            }

            Item {
                id: saveItem
                Layout.fillWidth: true
                width: btnSize
                height: 30
                //x: resetItem.x + resetItem.width + spaceSize
                anchors.right: loadItem.left
                anchors.rightMargin: spaceSize
                y: 5

                Rectangle {
                    id: saveHoverRect
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
                    id: saveText
                    color: "#ffffff"
                    text: "Save"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent
                }

                Rectangle {
                    id: saveRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id:  saveMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        visible: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                saveHoverRect.visible = true;
                            }else{
                                saveHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            onSaveLUT()
                        }
                    }
                }
            }

            Item {
                id: loadItem
                Layout.fillWidth: true
                width: btnSize
                height: 30
                //x: saveItem.x + saveItem.width + spaceSize
                anchors.right: parent.right
                anchors.rightMargin: spaceSize
                y: 5

                Rectangle {
                    id: loadHoverRect
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
                    id: loadText
                    color: "#ffffff"
                    text: "Load"
                    z: 1
                    font.pointSize: 10
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    anchors.fill: parent
                }

                Rectangle {
                    id: loadRect
                    width: parent.width
                    height: parent.height
                   anchors.fill: parent
                    color: "#373737"

                    MouseArea {
                        id:  loadMouseArea
                        width: 60
                        anchors.fill: parent
                        hoverEnabled: true
                        visible: true
                        onHoveredChanged: {
                            isHovered = !isHovered
                            if(isHovered){
                                loadHoverRect.visible = true;
                            }else{
                                loadHoverRect.visible = false;
                            }
                        }

                        onClicked: {
                            onOpenLUT()
                        }
                    }
                }
            }

            FileDialog {
                id: lutSaveDialog
                title: "Save LUT file"
                selectExisting: false
                selectMultiple: false
                selectFolder: false
                //folder: shortcuts.home
                nameFilters: ["LUT file (*.lut)"]
                selectedNameFilter: "All files(*)"

                onAccepted: {
                    var filePath = fileUrls.toString().substring(8)
                    qmlMainWindow.saveLUT(filePath)
                }
            }

            FileDialog {
                id: lutOpenDialog
                title: "Open LUT file"
                selectExisting: true
                selectMultiple: false
                selectFolder: false
                //folder: shortcuts.home
                nameFilters: ["LUT file (*.lut)"]
                selectedNameFilter: "All files(*)"

                onAccepted: {
                    var filePath = fileUrls.toString().substring(8)
                    qmlMainWindow.loadLUT(filePath)

                    //onLoadData()
                }
            }
        }

        Rectangle {
            id: lutScreen
            x: 0
            y: 50
            width: parent.width
            height: parent.width - 50
            color: "black"
            HiResItem {
                anchors.fill:  parent

                QChartJs {
                    id: chart_line
                    autoRedraw: true
                    anchors.fill: parent

                    chartType: ChartTypes.QChartJSTypes.LINE

                    //animation: true
                    chartAnimationEasing: Easing.InOutCubic;
                    chartAnimationDuration: 300;
                    chartOptions: ({scaleFontColor: "#DDDDDD",
                                        scaleLineColor:'#DDDDDD',
                                    scaleGridLineColor: '#808080'})
                    chartData: ({
                            labels: ["0.0", "0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
                            datasets: [{
                                    fillColor : "rgba(220,220,220,0.2)",
                                    strokeColor : "rgba(220,220,220,1)",
                                    pointColor : "rgba(220,220,220,1)",
                                    pointStrokeColor : "#fff",
                                    pointHighlightFill : "#fff",
                                    pointHighlightStroke : "rgba(220,220,220,1)",
                                    data : [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
                                }]
                        })

                    MouseArea{
                        anchors.fill: parent
                        onClicked:
                        {
                            chart_mouse_move_event()
                        }
                    }

                    onChartUpdate:
                    {
                        var lutData = [];
                        var lutDataSize = chartData.datasets[0].data.length;
                        var colorType = 0;

                        for (var i = 0; i < lutDataSize; i++) {
                            lutData.push(chartRenderHandler.datasets[0].points[i].value);
                        }

                        switch (lutType){
                            case "Gray":
                                colorType = 0;
                                grayLutData = lutData;
                                break;
                            case "Red":
                                colorType = 1;
                                redLutData = lutData
                                break;
                            case "Green":
                                colorType = 2;
                                greenLutData = lutData
                                break;
                            case "Blue":
                                colorType = 3
                                blueLutData = lutData
                                break;
                            default:
                                break;
                        }

                        qmlMainWindow.luTChanged(lutData, colorType);
                    }
                }
            }
        }
	}

    onFloatingChanged: chart_line.invalidate()
    onAnimationsActiveChanged: chart_line.invalidate()

	function chart_mouse_move_event(){
        chart_line.requestChangeDataPaint()
	}

    function updateLUT(lutData)
    {
        for (var i = 0; i < 11; i++) {
            chart_line.chartRenderHandler.updateChartData(lutData[i], i);
        }

        chart_line.requestPaint();
    }

    function onReset(lutType)
    {
		var colorType = 0;
        switch (lutType){
            case "Gray":
				colorType = 0;
                grayLutData = initLutData;
                break;
            case "Red":
				colorType = 1;
                redLutData = initLutData
                break;
            case "Green":
				colorType = 2;
                greenLutData = initLutData
                break;
            case "Blue":
				colorType = 3;
                blueLutData = initLutData
                break;
            default:
                break;
        }
        updateLUT(initLutData);
        qmlMainWindow.luTChanged(initLutData, colorType);
    }

    function onLoadData(lutData, colorType)
    {
		switch(colorType){
			case 0:
                grayLutData = lutData;
                if(lutType == "Gray"){
                    updateLUT(lutData);
                }
				break;
			case 1:
                redLutData = lutData
                if(lutType == "Red"){
                    updateLUT(lutData);
                }
				break;
			case 2:
                greenLutData = lutData
                if(lutType == "Green"){
                    updateLUT(lutData);
                }
				break;
			case 3:
                blueLutData = lutData
                if(lutType == "Blue"){
                    updateLUT(lutData);
                }
				break;
			default:
                break;

		}
    }    

    function setSelectedBtnColor (lutType) {
        switch (lutType){
            case "Gray":
                grayRect.color = selectedBtnColor
                redRect.color = "#373737"
                greenRect.color = "#373737"
                blueRect.color = "#373737"
                break;
            case "Red":
                grayRect.color = "#373737"
                redRect.color = selectedBtnColor
                greenRect.color = "#373737"
                blueRect.color = "#373737"
                break;
            case "Green":
                grayRect.color = "#373737"
                redRect.color = "#373737"
                greenRect.color = selectedBtnColor
                blueRect.color = "#373737"
                break;
            case "Blue":
                grayRect.color = "#373737"
                redRect.color = "#373737"
                greenRect.color = "#373737"
                blueRect.color = selectedBtnColor
                break;
            default:
                break;
        }
    }

    function onSaveLUT() {
        lutSaveDialog.visible = true
    }

    function onOpenLUT() {
        lutOpenDialog.visible = true
    }
}
