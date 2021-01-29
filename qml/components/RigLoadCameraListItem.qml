import QtQuick 2.5
import QtQuick.Controls 1.4

Item{   
    width: 300
    height: 50

    property string     title: "title"
    property int        index
    property bool       isManual: true
    property int        comIndex   
	property int        inputCount   

	Component.onCompleted:
		loadPASCameraList();

    Rectangle{
        id: textRec
        color: "#000000"
        anchors.fill: parent.fill
        height : parent.height
        anchors.margins: 4
        anchors.centerIn: parent.Center

        Text {
            id: cameraLabel
            anchors.left: parent.left
            anchors.leftMargin: 20
            anchors.verticalCenter: parent.verticalCenter
            text: qsTr(title)
            color: "#ffffff"
            font.pixelSize: 14
        }

        ComboBox {
            id: cameraCombo
            anchors.left: cameraLabel.right
            anchors.leftMargin: 20
            anchors.verticalCenter: parent.verticalCenter
            width: 180
            height: 30
            model: ListModel {
                id: pasCameraModel
            }
            onCurrentIndexChanged:{
                comIndex = currentIndex
                rigLoadBox.setPACCombo(index, comIndex)
            }
        }
    }

    function loadPASCameraList()
    {		
        pasCameraModel.clear();
		pasCameraModel.append({"text": "None"});
        for(var i = 0; i < getInputCount(); i++){
            pasCameraModel.append({"text": "Input" + (i + 1)});
        }
        
        setPASCameraList(comIndex);            
        
		cameraCombo.enabled = isManual;
    }

    function setPASCameraList(selIndex)
    {
        cameraCombo.currentIndex = selIndex
    }
}
