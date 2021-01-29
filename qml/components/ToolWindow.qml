import QtQuick 2.4
import QtQuick.Controls 1.4

FloatingWindow {
    //windowMenuGroup: 'tool'
	id: root_toolWindow
    property int        uiSpacing: 4
    property string     toolWindowTextID
    property bool       destroyOnClose: true
    property bool       isHovered: false
    detachable:         true
    signal closing()

    handle: Rectangle {
        color: "#000000"
        anchors.margins: 4
        anchors.centerIn: parent.Center
        height: 30

        Text {
            y: (30 - height) / 2
            color: "#ffffff"
            text: title
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.left: parent.left
            textFormat: Text.PlainText
            wrapMode: Text.WordWrap
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: fontSize
        }


        Row {
            id: r1
            anchors.verticalCenter: parent.verticalCenter
            anchors.right: parent.right
            spacing: uiSpacing

            Image {
                id: img3
                z: 1
                source: "/resources/icon-close-black.png"
                width: 45
                height: 30
                fillMode: Image.PreserveAspectFit
                anchors.verticalCenter: parent.verticalCenter

                Rectangle {
                    id: closeHoverRetangle
                    z: -1
                    width: 45
                    height: 30
                    color: "#c75050"
                    visible: false
                }
                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true

                    onEntered: closeHoverRetangle.visible = true;
                    onExited: closeHoverRetangle.visible = false;

                    onClicked: {
                        root_toolWindow.visible = false;
                        root.closeCalibSettingBox();
                        root.closeWeightMapSettingsBox();
                        qmlMainWindow.stopSingleCapture()
                        qmlMainWindow.finishSingleCameraCalibration()
                        liveBox.isDetailed = false;
                        if(toolbox.isWeightmap){
                            root.onCloseWeightmap();
                            statusItem.setPlayAndPause();
                        } else if (toolbox.isCTSetting) {
							root.onCloseCTSettingbox();
						} else if (toolbox.isStitchCameraTemplate) {
							root.onCloseStitchCameraTemplateSettingbox();
						}
                    }
                }
            }
        }
    }
}

