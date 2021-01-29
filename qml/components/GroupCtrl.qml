import QtQuick 2.0

Item {
    width: 255
    height: 48
    property bool isHoveredCheck: false
    property bool isHoveredUncheck: false
    property bool isHoveredMore: false
    property bool isHover: false

    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#1f1f1f"
    }

    ToolbarItem {
        id: snapshotCtrl
        anchors.right: spliter.left
        title: ""
        imgUrl: "../../resources/snapshot.png"
        theme: applicationSettingWindow.uiTheme
        autoHide: root.isHiddenToolbarItem
        fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"

        onClicked: {

        }
    }

    Item {
        id: spliter
        y: (parent.height - height) / 2
        width: 2
        height: parent.height - 20
        anchors.right: checkItem.left
        anchors.rightMargin: 15
        Rectangle{
            color: "#1f1f1f"
            x: 1
            height: parent.height
            width: 1

        }
        Rectangle{
            color: "#3f3f3f"
            x: 2
            height: parent.height
            width: 1


        }
    }

    ToolbarItem {
        id: checkItem
        anchors.right: uncheckItem.left
        title: ""
        imgUrl: "../../resources/check.png"
        theme: applicationSettingWindow.uiTheme
        autoHide: root.isHiddenToolbarItem
        fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"

        onClicked: {
            checkSnapshot();
        }
    }

    ToolbarItem {
        id: uncheckItem
        anchors.right: moreItem.left
        title: ""
        imgUrl: "../../resources/uncheck.png"
        theme: applicationSettingWindow.uiTheme
        autoHide: root.isHiddenToolbarItem
        fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"

        onClicked: {
            cancelSnapshot();
        }
    }

    ToolbarItem {
        id: moreItem
        anchors.right: parent.right
        title: ""
        imgUrl: "../../resources/more_control.png"
        theme: applicationSettingWindow.uiTheme
        autoHide: root.isHiddenToolbarItem
        fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"

        onClicked: {
            toolbox.showSpecifyBox()
        }
    }

    function checkSnapshot(){
        groupCtrl.state = "collapsed"
        groupCtrl.isHover = false;
        snapshotCtrl.visible = true;

		console.log("checkSnapshot");

        if(root.curMode === 1){
            qmlMainWindow.snapshotFrame();
            toolbox.initLiveTopControls();
        } else if(root.curMode === 2){
            qmlMainWindow.snapshotPanoramaFrame();
            toolbox.initSphericalTopControls();
			qmlMainWindow.snapshotPanoramaFrame();
        }

        statusItem.setPlayAndPause();

    }

    function cancelSnapshot(){
        groupCtrl.state = "collapsed"
        groupCtrl.isHover = false;
        snapshotCtrl.visible = true;
		cameraTemplateCtrl.visible = true;
		calibrationCtrl.visible = true;

        if(root.curMode === 1){
            toolbox.initLiveTopControls();
        } else if(root.curMode === 2){
            toolbox.initSphericalTopControls();
        }

        statusItem.setPlayAndPause();
    }


}

