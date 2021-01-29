import QtQuick 2.0

Item {
    id: weightmapSettingItem
    anchors.right: parent.right
    width: 96
    height: 48
    visible: false

    ToolbarItem {
                id: weightmapResetItem
                anchors.right: weightmapCloseItem.left
                imgUrl: ""
                title: "Reset"
                theme: applicationSettingWindow.uiTheme
                autoHide: root.isHiddenToolbarItem
                fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                onClicked: {
                    qmlMainWindow.onCancelCameraSettings();
                    qmlMainWindow.reStitch(true);

                    qmlMainWindow.resetWeightMap();
                }
            }

    ToolbarItem {
                id: weightmapCloseItem
                anchors.right: parent.left
                imgUrl: "../../resources/uncheck.png"
                title: ""
                theme: applicationSettingWindow.uiTheme
                autoHide: root.isHiddenToolbarItem
                fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"


                onClicked: {
					root.closeWeightMapSettingsBox();
                    root.onCloseWeightmap();
                    statusItem.setPlayAndPause();
                }
            }
}

