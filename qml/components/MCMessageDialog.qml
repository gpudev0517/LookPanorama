import QtQuick 2.5
import QtQuick.Controls 1.4 //2.0
import QtQuick.Controls.Styles 1.4
import QtQuick.Controls.Styles.Flat 1.0 as Flat
//import QtQuick.layouts 1.1
import "../"
import "."
import QtQuick.Layouts 1.1

import QtQuick.Window 2.2
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import "../controls"

ToolWindow{
    id:messageDlg
    width: 240
    height: 120
    z: 10
    windowState: qsTr("windowed")
    visible: false


    property string   title: "Calibration"
    property int      fontSize: 15
    property int      itemHeight: 30
    property int      spacing: 20
    property color    textColor: "#ffffff"
    property var      currentCursorShape: Qt.CrossCursor
    property int      lblWidth: 60
    property bool     isEditweightmap: true


    Item {
        id: itemMessage
        anchors.rightMargin: 0
        anchors.bottomMargin: 0
        anchors.leftMargin: 0
        anchors.topMargin: 0
        anchors.fill: parent

        Text {
            id: textMsgContent
            x: 8
            y: 8
            width: 224
            height: 33
            text: qsTr("Calibration Single Camera?")
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: 17
            color: textColor
        }

        Button {
            id: buttonConfirm
            x: 8
            y: 76
            width: 100
            height: 36
            text: qsTr("Confirm")
            z: 0
            checkable: false

            onClicked: {
                qmlMainWindow.applySingle()
                qmlMainWindow.finishSingleCameraCalibration()

                messageDlg.visible = false;

            }
        }

        Button {
            id: buttonCancel
            x: 132
            y: 76
            width: 100
            height: 36
            text: qsTr("Cancel")

            onClicked: {
                console.log("CalibrationCancelClicked:")
                qmlMainWindow.finishSingleCameraCalibration()

                messageDlg.visible = false;
            }
        }
    }

}
