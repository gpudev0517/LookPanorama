import QtQuick 2.5
import QtQuick.Controls 1.4

Item{
    width: 300
    height: 50

    property string     title: "title"
    property int        index
    property bool       isManual: true
    property bool       isChecked: false

    Component.onCompleted:
        savePASCameraList()

    Rectangle{
        id: textRec
        color: "#000000"
        anchors.fill: parent.fill
        height : parent.height
        anchors.margins: 4
        anchors.centerIn: parent.Center

        MCCheckBox {
            id: saveCheck
            width: 180
            height: 30
            anchors.left: parent.left
            anchors.leftMargin: 20
            anchors.verticalCenter: parent.verticalCenter
            text: qsTr(title)
            onCheckedChanged: {
                isChecked = checked
                rigSaveBox.setPACCombo(index, isChecked)
            }
        }
    }

    function savePASCameraList()
    {
		saveCheck.checked = true;
		saveCheck.enabled = isManual;
    }
}
