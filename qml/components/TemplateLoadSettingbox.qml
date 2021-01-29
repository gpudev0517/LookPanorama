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
    id : templateLoadBox
    property string   title: "Select Rig Input"
	property int      fontSize: 15
    visible: true    
    property int  index	
	property int  m_inputCount;		
    property int  m_type;
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
                initUI(m_inputCount);
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
                TemplateCameraListItem {
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
                    onLoad();
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
                    templateLoadBox.visible = false;
                }
            }
        }
    }

    function getCount() {
        return rigLoadList.count;
    }		

    function initUI(slotCount) {
        m_inputCount = slotCount;
        var isManual = !pasSwitch.checked;

        rigLoadList.clear();
        for(var i = 0; i < m_inputCount; i++){
            var selectIndex = i + 1;
            if (i >= m_inputCount)
                selectIndex = 0;
            rigLoadList.append({"titleText": "Camera" + (i + 1), "listIndex": i, "manualFlag": isManual, "comboIndex":selectIndex});
        }
    }

    function updateUI(type, curSlotList) {
        m_type = type;

        m_inputCount = curSlotList.length;
        rigLoadList.clear();

        var orderMapList = qmlMainWindow.getTemplateOrderMapList();

        if (orderMapList.length === 0) {
            pasSwitch.checked = true;
            initUI(m_inputCount);
            return;
        }

        // get slotList, and orderList from slotOrderMapList
        var slotList = [], orderList = [];
        var index = 0;
        for (var i = 0 ; i < orderMapList.length ; i++) {
            var strSlotOrderInfo = orderMapList[i].split(",");
            if (strSlotOrderInfo.length !== 2)
                continue;
            slotList[index] = strSlotOrderInfo[0];
            orderList[index] = strSlotOrderInfo[1] * 1;
            index++;    
        }

        // check switchStatus by comparing the orderList
        var isManual = false;
        for (var i = 0 ; i < orderMapList.length ; i++) {
            if (orderList[i] !== i || m_inputCount != orderList.length) {
                isManual = true;
                break;
            }
        }        
        pasSwitch.checked = !isManual;

        console.log("manual Information: slotCount: " + m_inputCount + " orderList.length: " + orderList.length + " isManual: " + isManual);

        // set the input to every camera        
        for (var i = 0 ; i < m_inputCount ; i++) {
            var selectIndex = i + 1;
            var curSlotName = curSlotList[i];
            // get the order by slot name
            for (var j = 0 ; j < slotList.length ; j++) {
                var slotName = slotList[j];
                if (curSlotName === slotName) {
                    selectIndex = orderList[j] + 1;
                    break;
                }
            }

            rigLoadList.append({"titleText": "Camera" + (i + 1), "listIndex": i, "manualFlag": isManual, "comboIndex":selectIndex});
        }        
    }

    function setPACCombo(l, c){
        rigLoadList.get(l).comboIndex = c
    }

    function getOrderList()
    {
        var orderList = [];
		if (root.isRigTemplateMode())
		{
			for(var i=0; i<getCount(); i++)
			{
				var currentListItem =  rigLoadList.get(i);
				var index = currentListItem.comboIndex;
				orderList.push(index - 1);
			}
		}
        return orderList;
    }

	function getSlotListByStringByType()
	{
		if (m_type == 1)
            return liveSettingbox.getSlotListByString();
        else if (m_type == 2)
            return videoSettingbox.getSlotListByString();
        else if (m_type == 3)
            return imageSettingbox.getSlotListByString();
	}

	function onLoad()
	{
		if (!root.isRigTemplateMode())
			return "";

		var strSlotList = getSlotListByStringByType();
		var strOrderList = getOrderList();
        qmlMainWindow.setTemplateOrder(strSlotList, strOrderList);
        templateLoadBox.visible = false;

        if (m_type == 1)
            liveSettingbox.updateTemplateSettingUI();
        else if (m_type == 2)
            videoSettingbox.updateTemplateSettingUI();
        else if (m_type == 3)
            imageSettingbox.updateTemplateSettingUI();
	}
}
