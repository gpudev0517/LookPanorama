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
    id : rigLoadBox
	property string   title: "Rig Template Loading"
	property int      fontSize: 15
    visible: true    
    property int  index	
	property var m_filePath
	property int  m_inputCount;	
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
                initUI();
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
        id: rigLoadListView
        width: 200
        height: 300
        anchors.top : headerItem.bottom
        anchors.topMargin: 20
        spacing : 40
        model: ListModel {
            id: rigLoadList
        }
        delegate: Item {
            width: parent.width
            Row {
                RigLoadCameraListItem {
                    title: titleText
                    index : listIndex
                    isManual: manualFlag
                    comIndex: comboIndex					
                }
            }
        }
    }

    Item {
        id: loadItem
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20
        anchors.leftMargin: 120
        width: 65
        height: 30

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
            z: 1
            color: "#ffffff"
            text: "Load"
            font.pointSize: 11
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
                id: loadMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true

                onEntered: loadHoverRect.visible = true
                onExited: loadHoverRect.visible = false

                onClicked: {
                    var result = [];
                    for(var i=0; i<getCount(); i++)
                    {
                        var currentListItem =  rigLoadList.get(i);
						var index = currentListItem.comboIndex;
						result.push(index - 1);
                    }
                    qmlMainWindow.loadTemplatePAC(m_filePath, result);
					m_filePath = "";
                    rigLoadBox.visible = false;
                }
            }
        }
    }

    Item {
        id: cancelItem
        anchors.left: loadItem.right
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
					m_filePath = "";
                    rigLoadBox.visible = false;
                }
            }
        }
    }

    function getCount() {
        return rigLoadList.count;
    }

	function setFilePath(filePath)
	{
		m_filePath = filePath;
		m_inputCount = qmlMainWindow.loadTemplatePAC(m_filePath, []);
		pasSwitch.checked = true;
		initUI();
	}

	function getInputCount() {		
		return m_inputCount;
	}

    function initUI() {			
		var isManual = !pasSwitch.checked;	
        rigLoadList.clear();
        for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
			var selectIndex = i + 1;
			if (i >= m_inputCount)
				selectIndex = 0;
            
			rigLoadList.append({"titleText": "Camera" + (i + 1), "listIndex": i, "manualFlag": isManual, "comboIndex":selectIndex});						

        }
    }
    function setPACCombo(l, c){
        rigLoadList.get(l).comboIndex = c
    }
}
