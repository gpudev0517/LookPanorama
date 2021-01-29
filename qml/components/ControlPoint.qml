import QtQuick 2.0

Item {
    property int    cpIndex: -1
    property int    xPos: 0
    property int    yPos: 0
	property int	cpSize: 9
	property int	lblWidth: 20
	property int	lblHeight: 15
	property int	lblSpace: 5
    property var	lblColor: "Red"
	/* // Point is "+" symbol.
    Text {
        id: cPoint
        x: xPos - width / 2
        y: yPos - height / 2
        width: 14
        height: 14
        text: qsTr("+")
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        font.pixelSize:14
        color: "white"
    }
	*/
	Rectangle {
		x: xPos - (cpSize-1)/2
		y: yPos
		width: cpSize
		height: 1
		color: "white"
	}

	Rectangle {
		x: xPos
		y: yPos - (cpSize-1)/2
		width: 1
		height: cpSize
		color: "white"
	}

    Rectangle {
        id: cpLabel
        x: xPos + lblSpace
        y: yPos + lblSpace
        width: lblWidth
        height: lblHeight
        //radius: 4
        border.color: "#000000"
        //color: "#ff0000"
        color: lblColor
		//color: colors[2]
        border.width: 1

        Text {
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            text: cpIndex
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            color: "white"
            anchors.verticalCenter: parent.verticalCenter
        }
    }
}

