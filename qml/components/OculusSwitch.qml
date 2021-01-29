import QtQuick 2.0
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4

Item {
    id: oculusItem
    width: 150
    height: 48

    property string theme: "Default"
    property color  textColor: "#d0e3ef"

    FontLoader {
        id: localFont
        source: "../../resources/font/MTF Base Outline.ttf"
    }
    
	Text {
        id: chessboardLabel
        y: (parent.height - height) / 2
        anchors.right: chessboardSwitch.left
        anchors.rightMargin: 15
        anchors.verticalCenter: parent.verticalCenter
        color: theme == "Default"?"white":textColor
        text: qsTr("Show Grid")
        font.pixelSize: 16

        font.family: theme == "Default"? "Tahoma": localFont.name
    }

	Switch{
        id: chessboardSwitch
        y: (parent.height - height) / 2
        anchors.right: parent.right
        anchors.rightMargin: 20
        width: 50
        height: 30
        checked: false

        onClicked: {
            qmlMainWindow.showChessboard(checked);
        }
    }    

}
