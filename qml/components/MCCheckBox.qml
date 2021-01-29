import QtQuick 2.0
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.4
CheckBox {
    width: 100
    height: 62
    style: CheckBoxStyle {
        indicator: Rectangle {
                        implicitWidth: 18
                        implicitHeight: 18
                        radius: 1
                        color: "#ffffff"
                        border.color: control.activeFocus ? "darkblue" : "#303030"
                        border.width: 2
                        Rectangle {
                            visible: control.checked
                            color: "#555"
                            border.color: "#333"
                            radius: 1;
                            anchors.margins: 4
                            anchors.fill: parent
                        }
                }
        spacing: 5
        label: Text {
            text: control.text
            verticalAlignment: Text.AlignVCenter
            color: "#ffffff"
            font.pixelSize: 14
        }
    }
}

