import QtQuick 2.0

Item {
    property int screenNum: 1
    property bool isSelected: false
    width: 28
    height: 20
    Rectangle {
        id: numberRect
        color: "#171717"
        border.color:  "#929292"
        implicitWidth: parent.width
        implicitHeight: parent.height
        //radius: 2

        Text {
            id: text
            anchors.centerIn: parent
            text: screenNum
            color: "white"
        }
    }

    Rectangle {
        id: selectRect
        color: "#4e8d15"
        border.color:  "#929292"
        implicitWidth: parent.width
        implicitHeight: parent.height
        //radius: 2
        visible: isSelected

        Text {
            anchors.centerIn: parent
            text: screenNum
            color: "white"
        }
    }

    MouseArea{
        anchors.fill: parent
        hoverEnabled: true
        onClicked: {
            sphericalWindow.visible = false;

            for(var i = 0; i<numberList.count; i++){
                var item = numberList.get(i);
                item.selected = false;
            }

            var number = numberList.get(screenNum - 1);
            number.selected = true;

            var strRect = qmlMainWindow.getFullScreenInfoStr(screenNum - 1);
            var strRectList = strRect.split(":");

            sphericalWindow.createSphericalWindow(screenNum);

			qmlMainWindow.setFullScreenStitchViewMode(screenNum - 1);

            sphericalWindow.setX(strRectList[0]);
            sphericalWindow.setY(strRectList[1]);
            sphericalWindow.width = strRectList[2];
            sphericalWindow.height = strRectList[3];
        }
    }
}
