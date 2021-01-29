import QtQuick 2.5
import QtQuick.Dialogs 1.2
import QtQuick.Extras 1.4
import "components"
import QmlRecentDialog 1.0
import Pixmap 1.0

Item {
    id: recent
    width: 1030
    height: 688

    property bool       isHovered: false
    property int        minWidth: 1080
    property int        minHeight: 720
    property string     imagepath: ""    

	property var		isNextConfigurationQueued: false
    property int        selectedRecentIndex: 0    

	property string     m_selectedTemplateIniPath: ""

	Pixmap {
	   id: pix
	}

    Component.onCompleted:
    {
        initFavorite_TemplateGridList();
    }

    Text {
        id: recentTitleText
        x: 35
        y: 12
        color: "#ffffff"
        text: qsTr("Recent")
        z: 2
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        font.pixelSize: 22
    }

    Rectangle {
        id: titleRectangle
        width: parent.width
        height: 48
        color: "#1f1f1f"
        z: 1
    }

    Item {
        id: recentListItem
        width: 350
        height: parent.height
        z: 1
        Rectangle {
            id: recentListRectangle
            width: parent.width
            height: parent.height
            color: "#171717"
        }
        MouseArea {
            anchors.fill: parent
        }

        ListView {
            id: recentListView
            x: 15
            y: recentTitleText.y + recentTitleText.height + 20
            width: parent.width - recentTitleText.x
            height: count < 11 ? 58 * count : parent.height - 150
            spacing: 10
            model: ListModel {
                id: recentList

            }
            delegate: Item {
                x: 5
                width: parent.width
                height: 50
                Row {
                    RecentListitem {
                        title: titleText
                        iconPath: imagePath
                        path: sourcePath
                        fullPath: orgPath
                    }
                }
            }

            remove: Transition {
                   ParallelAnimation {
                       NumberAnimation { properties: "x"; to: -500; duration: 300 }
                   }
            }
        }

        Rectangle {
            id: openItemBackRectangle
            y: openItem.y
            width: parent.width
            height: parent.height - recentListView.height;
            color: "#171717"
        }

        Item {
            id: openItem
            x: recentTitleText.x
            z: 2
            width: parent.width - x
            height: 50

            Rectangle {
                id: spliterRectangle
                z: 2
                width: parent.width - 20
                height: 3
                color: "#1f1f1f"
            }

            anchors {
                top: recentListView.bottom
                topMargin: 15
            }



            Rectangle {
                id: openhoveredRectangle
                x: openText.x - 1
                y: openText.y + openText.height + 2
                z: 0
                width: openText.width + 2
                height: 1
                color: "#ffffff"
                visible: false
            }

            Image {
                id: openIconImage
                //x: 19
                y: (parent.height - height) / 2
                z: 1
                width: 28
                height: 28
                fillMode: Image.PreserveAspectFit
                source: "../resources/icon_open.png"
            }

            Text {
                id: openText
                anchors {
                    left: openIconImage.right
                    leftMargin: 10
                }

                y: (parent.height - height) / 2
                z: 1
                color: "#ffffff"
                text: qsTr("Open Project")
                font.pixelSize: 14
            }

            MouseArea {
                id: mouseArea
                x: 0
                z: 2
                width: openIconImage.width + openText.width
                height: parent.height
                antialiasing: true
                hoverEnabled: true
                onHoveredChanged: {
                    isHovered = !isHovered

                    if(isHovered) {
                        openhoveredRectangle.visible = true;
                        mouseArea.cursorShape = Qt.PointingHandCursor;
                    }
                    else {
                        openhoveredRectangle.visible = false;
                        mouseArea.cursorShape = Qt.ArrowCursor;
                    }
                }
                onClicked: {
                    root.onFileOpen()
                }
            }
        }
    }

    Rectangle {
        id: rightRectangle
        anchors.top: parent.top
        x: recentListItem.width
        width: parent.width
        height: parent.height
        color: "#1f1f1f"

        Item {
            id: newSetupItem
            width: parent.width - recentListItem.width
            height: parent.height / 3

            Text {
                id: newSetupTitle
                width: parent.width
                x: 30
                y: 55
                color: "#ffffff"
                font.pixelSize: 20
                text: qsTr("NEW")
            }

            GridView {
                id: newSetupGridView
                x: 43
                anchors.top: newSetupTitle.bottom
                width: parent.width - x
                height: parent.height - y
                contentWidth: 0
                cellHeight: 200
                model: ListModel {
                    ListElement {
                        titleText: "Live Camera"
                        //imagePath:"../resources/icon_camera.png"
                        imagePath:"../resources/icon_tempCamera.png"
                        selectType: 1

                    }

                    ListElement {
                        titleText: "Video"
                        imagePath: "../resources/icon_video.png"
                        //imagePath: "../resources/icon_temp_video.png"
                        selectType: 2
                    }

                    ListElement {
                        titleText: "Frame Sequence"
                        imagePath: "../resources/icon_image.png"
                        selectType: 3
                    }
                }
                delegate: Item {
                    x: 5
                    width: 60
                    height: 320
                    Column {
                        RecentGriditem {
                            title: titleText
                            iconPath: imagePath
                            type: selectType
                        }
                    }
                }
                cellWidth: 200
            }
        }

        Item {
            id: templateItem
            width: parent.width - recentListItem.width
            height: parent.height / 3
            anchors.top: newSetupItem.bottom

            Text {
                id: templateTitle
                width: parent.width
                x: 30
                y: 55
                color: "#ffffff"
                font.pixelSize: 20
                text: qsTr("TEMPLATES")
            }

            GridView {
                id: templateGridView
                x: 43
                anchors.top: templateTitle.bottom
                width: parent.width - x
                height: parent.height - y
                contentWidth: 0
                model: ListModel {
                    id: m_templateList
                }
                delegate: Item {
                    x: 5
                    width: 60
                    height: 320
                    Column {
                        TemplateGriditem {
                            title: titleText
                            iconPath: imagePath
                            index: iniIndex
                            iniPath: iniFilePath
                            isSelected: isItemSelected
                        }
                    }
                }
                cellWidth: 200
            }
        }

        Item {
            id: favoriteItem
            width: parent.width - recentListItem.width
            height: parent.height / 3
            anchors.top: templateItem.bottom

            Text {
                id: favoriteTitle
                width: parent.width
                x: 30
                y: 55
                color: "#ffffff"
                font.pixelSize: 20
                text: qsTr("FAVORITES")
            }

            GridView {
                id: favoriteGridView
                x: 43
                anchors.top: favoriteTitle.bottom
                width: parent.width - x
                height: parent.height - y
                contentWidth: 0
                cellHeight: 200
                model: ListModel {
                    id: m_favoriteList
                }
                delegate: Item {
                    x: 5
                    width: 60
                    height: 320
                    Column {
                        TemplateGriditem {
                            title: titleText
                            iconPath: imagePath
                            index: iniIndex
                            iniPath: iniFilePath
                            isSelected: isItemSelected
                        }
                    }
                }
                cellWidth: 200
            }
        }        
    }

	function setData(path){
		pix.load(path)
	}

    function initFavorite_TemplateGridList() {
        m_selectedTemplateIniPath = "";        

        // template list
        m_templateList.clear();
        var templateList = qmlMainWindow.loadTemplateList();
        for (var i = 0 ; i < templateList.length ; i++)
        {
            var list = templateList[i].split("/");
            //var fileName = list[list.length - 1];
			var fileName = qmlMainWindow.getDisplayName(templateList[i]);
			setData(templateList[i]);
			pix.setCurIndex(i);
            m_templateList.append({"titleText": fileName, "imagePath": pix.data,
                                   "iniIndex": i, "iniFilePath":templateList[i], "isItemSelected": false});
        }

        // favorite list
        m_favoriteList.clear();
        var favoriteList = qmlMainWindow.loadFavoriteTemplate();
        for (var i = 0 ; i < favoriteList.length ; i++)
        {
            var list = favoriteList[i].split("/");
            //var fileName = list[list.length - 1];
			var fileName = qmlMainWindow.getDisplayName(favoriteList[i]);
			setData(favoriteList[i]);
			pix.setCurIndex(templateList.length+i);
            m_favoriteList.append({"titleText": fileName, "imagePath": pix.data,
                                   "iniIndex": (templateList.length + i), "iniFilePath":favoriteList[i], "isItemSelected": false});
		}
    }

    function appendFavoriteItem(filePath)
    {
        var list = filePath.split("/");
        var fileName = list[list.length - 1];

        m_favoriteList.append({"titleText": fileName, "imagePath": "../resources/icon_tempCamera.png",
                               "iniIndex": m_favoriteList.length, "iniFilePath":filePath, "isItemSelected": false});
    }

    function selectFavorite_Template(index, iniPath, isSelected) {        
        m_selectedTemplateIniPath = isSelected ? iniPath : "";        

        var favoriteList = qmlMainWindow.loadFavoriteTemplate();
        var templateList = qmlMainWindow.loadTemplateList();

        m_favoriteList.clear();
        m_templateList.clear();

        for(var i = 0; i < favoriteList.length + templateList.length; i++){
            var cur_isSelected = false;
            if (i === index)
            {
                cur_isSelected = isSelected;
            }

            if (i < templateList.length)
            {                
                var list = templateList[i].split("/");
				var fileName = qmlMainWindow.getDisplayName(templateList[i]);
              //var fileName = list[list.length - 1];
				pix.setCurIndex(i);
                m_templateList.append({"titleText": fileName, "imagePath": pix.data,
                                       "iniIndex": i, "iniFilePath":templateList[i], "isItemSelected": cur_isSelected});
            } else {
                var favoriteIndex = (i - templateList.length);
                var list = favoriteList[favoriteIndex].split("/");
				var fileName = qmlMainWindow.getDisplayName(favoriteList[favoriteIndex]);
                //var fileName = list[list.length - 1];
				pix.setCurIndex(i);
                m_favoriteList.append({"titleText": fileName, "imagePath": pix.data,
                                       "iniIndex": i, "iniFilePath":favoriteList[favoriteIndex], "isItemSelected": cur_isSelected});
            }
        }
    }


    MouseArea {
        id: resizeMouseArea
        width: 5
        height: 5
        y: parent.height - height
        x: parent.width - width

        hoverEnabled: true
        onHoveredChanged: {
            if(!root.isFullScreen)
                cursorShape = Qt.SizeFDiagCursor
        }

        property bool isMoveMode: false
        property var clickPos: "1,1"
        property real movedX: 0
        property real movedY: 0

        onPressed: {
            if(mouse.button == Qt.LeftButton && !root.isFullScreen) {
                isMoveMode = true
                clickPos  = mapToItem(statusItem.parent, mouse.x,mouse.y)
            }
        }

        onPositionChanged: {
            if(isMoveMode) {
                var x = mouse.x
                var y = mouse.y
                var parentPos = mapToItem(statusItem.parent, x, y)
                var delta = Qt.point(parentPos.x-clickPos.x - movedX, parentPos.y-clickPos.y - movedY)
                if (root.width + delta.x >= minWidth) {
                    movedX += delta.x
                    root.width += delta.x
                }
                if (root.height + delta.y >= minHeight) {
                    movedY += delta.y
                    root.height += delta.y
                }
            }
        }

        onReleased: {
            if(mouse.button == Qt.LeftButton) {
                isMoveMode = false
                movedX = 0
                movedY = 0
            }
        }
    }

    Rectangle {
        id: resizeLine
        width: 30
        height: 2
        x: parent.width - width + 4
        y: parent.height - height - 3
        z: 5
        color: "#555555"
        rotation: -45
    }

    Rectangle {
        id: resizeLine2
        width: 20
        height: 2
        x: parent.width - width + 4
        y: parent.height - height - 3
        z: 5
        color: "#555555"
        rotation: -45
    }

    VideoSettingbox {
        id: videoSettingbox
        x: parent.width - width
        z: 2
        height: parent.height
        state: "collapsed"
        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: videoSettingbox
                    width: root.width
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: videoSettingbox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: videoSettingbox; property: "width"; duration: 100 }
                NumberAnimation { target: videoSettingbox; property: "opacity"; duration: 100 }
            }
        ]
    }

    ImageSettingbox {
        id: imageSettingbox
        x: parent.width - width
        z: 2
        height: parent.height
        state: "collapsed"
        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: imageSettingbox
                    width: root.width
                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: imageSettingbox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: imageSettingbox; property: "width"; duration: 100 }
                NumberAnimation { target: imageSettingbox; property: "opacity"; duration: 100 }
            }
        ]


    }

    DshowSettingbox {
        id: dshowSettingbox
        x: parent.width - width
        z: 2
        height: parent.height
        state: "collapsed"
        states: [
            State {
                name: "expanded"
                PropertyChanges {
                    target: dshowSettingbox
                    width: root.width

                }
            },
            State {
                name: "collapsed"
                PropertyChanges {
                    target: dshowSettingbox
                    width: 0
                }
            }
        ]

        transitions: [
            Transition {
                NumberAnimation { target: dshowSettingbox; property: "width"; duration: 100 }
                NumberAnimation { target: dshowSettingbox; property: "opacity"; duration: 100 }
            }
        ]

    }

	function getCount() {
		return recentList.count;
	}

	function addTop(index, isSort) {
        if (isSort){
			recentList.remove(index);
        }
        var iniName;
		var fullTitle = qmlMainWindow.getRecentFullPath(0);
        var title = qmlMainWindow.getDisplayName(fullTitle);
        if(title.length > 23){
            iniName = title.substring(0,20) + "..."
        } else {
            iniName = title;
        }

		var fullPath = qmlMainWindow.getRecentFullPath(0);
		var shortPath = qmlMainWindow.getRecentPath(0); 
		var type = qmlMainWindow.getRecentType(0);
		var imagepath = "../resources/icon_camera_small.png";
		
		switch(type)
		{
			case 0:
			case 1:
				imagepath = "../resources/icon_camera_small.png";
				break;
			case 2:
				imagepath = "../resources/icon_video_small.png";
				break;
			case 3:
				imagepath = "../resources/icon_image_small.png";
				break;
			default:
				break;
		}

        recentList.insert(0,{"titleText": iniName, "imagePath": imagepath, "sourcePath": shortPath, "orgPath": fullPath});
	}

	function append(title, fullPath, shortPath, type) {
        var iniName;
		title = qmlMainWindow.getDisplayName(fullPath);
        if(title.length > 23){
            iniName = title.substring(0,20) + "..."
        } else {
            iniName = title;
        }

		var imagepath = "../resources/icon_camera_small.png";
		switch(type)
		{
			case 0:
			case 1:
				imagepath = "../resources/icon_camera_small.png";
				break;
			case 2:
				imagepath = "../resources/icon_video_small.png";
				break;
			case 3:
				imagepath = "../resources/icon_image_small.png";
				break;
			default:
				break;
		}

        recentList.append({"titleText": iniName, "imagePath": imagepath, "sourcePath": shortPath, "orgPath": fullPath});
	}

	function deleteRecentItem(recentIndex){
        var deleteObject = recentList.get(recentIndex)
        qmlMainWindow.deleteRecentList(deleteObject.titleText)
        recentList.remove(recentIndex)
        qmlMainWindow.saveRecentMgrToINI()
    }

	function openRecentConfiguration(recentIndex)
	{
		selectedRecentIndex = recentIndex;
        //root.openCameraViews();
		if (qmlMainWindow.isEmpty)
		{
			onOpenProject();
		}
		else
		{
			isNextConfigurationQueued = true;
			toolbox.closeProject();
		}
	}

	function previousConfigurationClosed()
	{
		dshowSettingbox.previousConfigurationClosed();
		videoSettingbox.previousConfigurationClosed();
		imageSettingbox.previousConfigurationClosed();
		if (isNextConfigurationQueued)
		{
			isNextConfigurationQueued = false;
			onOpenProject();
		}
	}

	function onOpenProject()
	{        
		var selectedRecentPath = recentList.get(selectedRecentIndex).orgPath;
		var recentIndex = qmlMainWindow.recentOpenIniPath(selectedRecentPath);

        if(recentIndex === -1 || recentIndex === -2){
            recent.deleteRecentItem(selectedRecentIndex);
            return;
        }
        else if(recentIndex >= 0)
            root.addRecentList(recentIndex);
        
		initFavorite_TemplateGridList();

        root.initVariants();
        root.initUI();

		qmlMainWindow.openProject();
		root.createCameraViews();

		var title = qmlMainWindow.getRecentFullPath(0);
        root.setCurrentTitle(qmlMainWindow.getDisplayName(title));

        toolbox.clearSelected();

		
	   root.panoMode = 2;
       root.onChangePanoMode();
    }
}
