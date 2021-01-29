import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.2

ToolWindow{
    id: details
    width: 730
    height: 445
    z: 10
    windowState: qsTr("windowed")
    visible: false
    floatingWindowBorderColor: "#1f1f1f"
    property bool   isHover: false
    property int    camIndex1: 0
    property int    camIndex2: 0
    property int    cPointCount
    property string pos
    property var colors: ["Red","Green","Blue","Dark cyan","Magenta","#808000","Dark gray","Dark red","Dark green","Dark blue","Dark magenta","Gray","Light gray"]
    property string title: "Details"
    property int    fontSize: 20
    property var cpList1: []
    property var cpList2: []
    property int componentIndex: -1
	
    TabView {
        id: firstTab
        x: 10
        y: 15
        width: 350
        height: 380
        onCurrentIndexChanged: {
            camIndex1 = currentIndex;
            updatePreview();
            createCPoint();
        }
        ListView {
            id: firstPreview
            x: 0
            anchors.fill: parent

			CameraPreview{
				id: firstViewObject
				x: 0
				y: 0
			}
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
                color: "#171717"
                opacity: 0.9
            }
        }

        Rectangle {
            id: firstBackground
            anchors.fill: parent
            opacity: 0.0
        }
    }

    TabView {
        id: secondTab
        x: parent.width / 2 + 5
        y: 15
        width: 350
        height: 380
        onCurrentIndexChanged: {
            camIndex2 = currentIndex;
            updatePreview();
            createCPoint();
        }

        ListView {
            id: secondPreview
            y: 10
            anchors.fill: parent

			CameraPreview{
				id: secondViewObject
				x: 0
				y: 0
			}
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
                    anchors.centerIn: parent
                    text: styleData.title
                    color: styleData.selected ? "white" : "white"
                }
            }
            frame: Rectangle {
                color: "#171717"
                opacity: 0.9
            }
        }

        Rectangle {
            id: secondBackground
            anchors.fill: parent
            opacity: 0.0

        }
    }

    function createTab(){
        for(var i = 0; i < firstTab.count;)
        {
            firstTab.removeTab(0)
        }
        for(var i = 0; i < secondTab.count;)
        {
            secondTab.removeTab(0)
        }
        var cameraCnt = qmlMainWindow.getCameraCount();
        for(var i = 0; i < cameraCnt; i++){
            firstTab.addTab(i + 1);
            secondTab.addTab (i + 1);
        }

        liveBox.isDetailed = true;
     }

    function updatePreview()
    {
        qmlMainWindow.updateCameraView(firstViewObject.camView,firstTab.currentIndex)
        qmlMainWindow.updateCameraView(secondViewObject.camView,secondTab.currentIndex)
    }

    function clearTab()
    {
        var cameraCnt = qmlMainWindow.getCameraCount();
        for(var i = cameraCnt; i > 0; i--){
            firstTab.removeTab(i - 1);
            secondTab.removeTab (i -1);
        }
    }

    function getCPointCount()
    {
        cPointCount = qmlMainWindow.getCPointCount(camIndex1,camIndex2);
    }

    function createCPoint()
    {
        var curWidth = 350;
        var curHeight = 350;
        var orgWidth = qmlMainWindow.getTempWidth();
        var orgHeight = qmlMainWindow.getTempHeight();
        if(orgHeight/orgWidth > 1)
        {
            curHeight = 350;
            curWidth = curHeight * (orgWidth / orgHeight);
        }
        else
        {
            curWidth = 350;
            curHeight = curWidth * (orgHeight / orgWidth);
        }

        //var curHeight = curWidth * (orgHeight/orgWidth);
        var cPointCount = 0;
        cPointCount = qmlMainWindow.getCPointCount(firstTab.currentIndex,secondTab.currentIndex);

        clearAllCPoint1();
        clearAllCPoint2();

        componentIndex = -1;

        for (var i = 0; i < cPointCount; i++)
        {
            pos = qmlMainWindow.getCPoint(i, firstTab.currentIndex, secondTab.currentIndex);
            var posList = pos.split(":");
            var xPos1 = (curWidth/orgWidth) * posList[0] + (350-curWidth)/ 2;
            var yPos1 = (curHeight/orgHeight) * posList[1] + (350 - curHeight) / 2;
            var xPos2 = (curWidth/orgWidth) * posList[2] + (350 - curWidth) / 2 ;
            var yPos2 = (curHeight/orgHeight) * posList[3] + (350 - curHeight) / 2;
            var component = Qt.createComponent("ControlPoint.qml");

            if (component.status !== Component.Ready) continue;
            var cpoint1 = component.createObject(firstPreview, {"xPos": xPos1, "yPos": yPos1, "cpIndex": i, "lblColor": colors[i%colors.length]});
            var cpoint2 = component.createObject(secondPreview, {"xPos": xPos2, "yPos": yPos2,"cpIndex": i, "lblColor": colors[i%colors.length]});
            componentIndex++;
            cpList1[componentIndex] = cpoint1;
            cpList2[componentIndex] = cpoint2;

        }
    }

    function clearAllCPoint1(){
        if ( componentIndex == -1 ) return;
        for (var i = 0; i < cpList1.length; i++)
        {
            var component = cpList1[i];
            if (typeof(component.destroy) == "function")
                component.destroy();
        }
    }

    function clearAllCPoint2(){
        if ( componentIndex == -1 ) return;
        for (var i = 0; i < cpList2.length; i++)
        {
            var component = cpList2[i];
            if (typeof(component.destroy) == "function")
                component.destroy();
        }
    }
}

