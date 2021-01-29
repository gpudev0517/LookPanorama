import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.0

ToolWindow {
    id: applicationSettingWindow
    width: 400
    height: 500
    z: 10
    //windowState: qsTr("windowed")
    visible: false
    //color: "#1f1f1f"
    //border.color: "green"

    property string   title: "Application Setting"
    property int      fontSize: 15
    property int      itemHeight: 30
    property int      spacing: 20
    property color    textColor: "#ffffff"
    property var      currentCursorShape: Qt.CrossCursor
    property int      lblWidth: 60
    property bool     isEditweightmap: true
    property CameraParamsSettingbox cameraParamsSettingbox;
    property int      m_weightMapMode: 1

    property string      uiTheme: "Default"

    property bool        isChangedUseCUDA: false
    property bool        originUseCUDA
    property int         originUiThemeIndex

    Component.onCompleted: {
        originUseCUDA = qmlMainWindow.applicationSetting.useCUDA
        originUiThemeIndex = getCurrentIndex(qmlMainWindow.applicationSetting.theme)
    }

    property int         btnSize: 95

    Item {
        id: useCUDAItem
        width: parent.width
        height: itemHeight
        anchors.topMargin: spacing
        y: 20
        Text {
            id: labelText
            color: textColor
            width: lblWidth
            text: qsTr("Use CUDA")
            horizontalAlignment: Text.AlignRight
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }

        Switch{
            id: drawMethodSwitch
            anchors.left: labelText.right
            anchors.leftMargin: spacing
            width: 50
            height: 30
            checked: qmlMainWindow.applicationSetting.useCUDA
            enabled: qmlMainWindow.applicationSetting.isCudaAvailable
            onClicked: {
                qmlMainWindow.applicationSetting.setUseCUDA(drawMethodSwitch.checked)

                if (originUseCUDA != drawMethodSwitch.checked)
                    isChangedUseCUDA = true
                else
                    isChangedUseCUDA = false
            }
        }

        Text {
            id: tooltipText
            color: textColor
            width: lblWidth
            text: qsTr("This setting will be applied from \n the next configuration.")
            horizontalAlignment: Text.AlignLeft
            anchors.top: drawMethodSwitch.bottom
            anchors.topMargin: spacing
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.right: parent.right
            anchors.rightMargin: spacing
            font.pixelSize: 12
            visible: isChangedUseCUDA
        }
     }

    Item {
        id: uiThemeItem
        width: parent.width
        height: 30
        anchors.top: useCUDAItem.bottom
        anchors.topMargin: isChangedUseCUDA? 10 + spacing + tooltipText.paintedHeight: spacing

        Text {
            id: uiThemeText
            color: textColor
            width: lblWidth
            text: qsTr("UI Theme")
            horizontalAlignment: Text.AlignLeft
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 13
        }

        ComboBox {
            id: uiThemeCombo
            width: parent.width / 2
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: uiThemeText.right
            anchors.leftMargin: spacing
            model: ["Default", "VRSE"]
            enabled: true

            currentIndex:  getCurrentIndex(qmlMainWindow.applicationSetting.theme)

            onCurrentTextChanged: {
                qmlMainWindow.applicationSetting.setTheme(currentText)
                uiTheme = currentText
            }
        }
    }

    Item {
        id: groupItem
        width: parent.width
        height: 40
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 0

        Item {
            id: applyItem

            Layout.fillWidth: true
            width: btnSize
            height: 30
            x: spacing - 10
            y: 5

            Rectangle {
                id: applyHoverRect
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
                id: applyText
                color: "#ffffff"
                text: "Apply"
                z: 1
                font.pointSize: 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                anchors.fill: parent
            }

            Rectangle {
                id: applyRect
                width: parent.width
                height: parent.height
               anchors.fill: parent
                color: "#373737"

                MouseArea {
                    id:  applyMouseArea
                    width: 60
                    anchors.fill: parent
                    hoverEnabled: true
                    visible: true
                    onHoveredChanged: {
                        isHovered = !isHovered
                        if(isHovered){
                            applyHoverRect.visible = true;
                        }else{
                            applyHoverRect.visible = false;
                        }
                    }

                    onClicked: {
                        qmlMainWindow.applicationSetting.saveApplicationSetting()
                        originUseCUDA = drawMethodSwitch.checked
                        originUiThemeIndex = uiThemeCombo.currentIndex
                        isChangedUseCUDA = false

                        applicationSettingWindow.visible = false
                    }
                }
            }
        }

        Item {
            id: cancelItem

            Layout.fillWidth: true
            //Layout.fillHeight: true
            width: btnSize
            height: 30
            x: spacing - 2 + applyItem.width
            y: 5

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
                color: "#ffffff"
                text: "Cancel"
                z: 1
                font.pointSize: 10
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
                    id:  cancelMouseArea
                    width: 60
                    anchors.fill: parent
                    hoverEnabled: true
                    visible: true
                    onHoveredChanged: {
                        isHovered = !isHovered
                        if(isHovered){
                            cancelHoverRect.visible = true;
                        }else{
                            cancelHoverRect.visible = false;
                        }
                    }

                    onClicked: {
                        drawMethodSwitch.checked = originUseCUDA
                        uiThemeCombo.currentIndex = originUiThemeIndex

                        applicationSettingWindow.visible = false
                    }
                }
            }
        }
    }

    function getCurrentIndex(string) {
        if (string == "Default")
            return 0
        else if (string == "VRSE")
            return 1
    }
}
