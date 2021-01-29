import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2
import "../controls"

ToolWindow{
    id : rigSaveBox
	property string   title: "Rig Template Save"
	property int      fontSize: 15
    visible: true    
    property bool isChecked: true
    property int  index
	property var m_filePath
	z: 11
    

    Item {
        id: headerItem
        width: parent.width
        height: 30
        anchors.top: parent.top
        anchors.topMargin: 20
        anchors.horizontalCenter: parent.horizontalCenter
        Text {
            id: manualLabel
            x:parent.width * 0.2
            color: "#ffffff"
            text: qsTr("Manual")
            horizontalAlignment: Text.AlignRight
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 14

        }

        Switch{
            id: pasSwitch
            anchors.left: manualLabel.right
            anchors.leftMargin: 20
            width: 50
            height: 30
            checked: true
            onClicked: {                
                initUI()
            }
        }

        Text {
            id: autoLabel
            color: "#ffffff"
            text: qsTr("Auto")
            horizontalAlignment: Text.AlignLeft
            anchors.left: pasSwitch.right
            anchors.leftMargin: 20
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 14
        }
    }

    ListView {
        id: rigSaveListView
        width: 200
        height: 300
        anchors.top : headerItem.bottom
        anchors.topMargin: 20
        spacing : 40
        model: ListModel {
            id: rigSaveList

        }
        delegate: Item {
            width: parent.width
            Row {
                RigSaveCameraListItem {
                    title: titleText
                    index : listIndex
                    isManual: manualFlag
                    isChecked : checkFlag
                }
            }
        }
    }

    Item {
        id: saveItem
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20
        anchors.leftMargin: 120
        width: 65
        height: 30

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
            z: 1
            color: "#ffffff"
            text: "Save"
            font.pointSize: 11
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
                id: saveMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true

                onEntered: saveHoverRect.visible = true
                onExited: saveHoverRect.visible = false

                onClicked: {
                    var result = [];
                    for(var i=0; i<getCount(); i++)
                    {
                        var currentListItem =  rigSaveList.get(i);
                        if(currentListItem.checkFlag === true)
						{
							result.push(i);
						}
                    }
                    qmlMainWindow.saveTemplatePAC(m_filePath, result);
                    rigSaveBox.visible = false;
                }
            }
        }
    }

    Item {
        id: cancelItem
        anchors.left: saveItem.right
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20
        anchors.leftMargin: 20
        width: 65
        height: 30

        Rectangle {
            id: cancelHoverRect
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
            id: cancelText
            z: 1
            color: "#ffffff"
            text: "Cancel"
            font.pointSize: 11
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.fill: parent
        }

        Rectangle {
            id: cancelRect
            width: parent.width
            height: parent.height
           anchors.fill: parent
            color: "#373737"

            MouseArea {
                id: cancelMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true

                onEntered: cancelHoverRect.visible = true
                onExited: cancelHoverRect.visible = false

                onClicked: {
                    rigSaveBox.visible = false
                }
            }
        }
    }

    function getCount() {
        return rigSaveList.count;
    }

	function setFilePath(filePath)
	{
		m_filePath = filePath;
		pasSwitch.checked = true;
		initUI();
	}

    function initUI() {
		var isManual = !pasSwitch.checked;
        rigSaveList.clear();
        for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
            if(isManual == true){
                rigSaveList.append({"titleText": "Camera" + (i + 1), "listIndex": i, "manualFlag": isManual, "checkFlag": isChecked});
            }else{
                rigSaveList.append({"titleText": "Camera" + (i + 1), "listIndex": i, "manualFlag": isManual, "checkFlag": isChecked});
            }
        }
    }

    function setPACCombo(l, c){        
        rigSaveList.get(l).checkFlag = c
    }
}
