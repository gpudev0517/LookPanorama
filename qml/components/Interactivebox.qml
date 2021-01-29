import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Layouts 1.1
import "../controls"

Item {
    width : 350
    height: 684
    opacity: 1


    property int        hoveredType: 0
    property bool       isHovered : false
    property bool       isSelected: false
    property int        leftMargin: 20
    property int        rightMargin: 20
    property int        lblFont: 14

    Rectangle {
        id: titleRectangle
        x: 0
        y: 0
        width: parent.width
        height: 48
        color: "#171717"
        z: 1
        opacity: 1

        Text {
            id: titleText
            x: (250 - width) / 2
            y: (parent.height - height) / 2
            z: 3
            color: "#ffffff"
            text: qsTr("Interactive")
            font.bold: false
            font.pixelSize: 20
        }
    }

    Rectangle {
        id: spliterRectangle
        width: parent.width
        height: 2
        z: 3
        anchors.top: titleRectangle.bottom
        color: "#1f1f1f"
    }

    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#171717"
        opacity: 0.9
    }

    Item {
        id: oculusRiftItem
        width: 350
        height: 30
        anchors.top: titleRectangle.bottom
        anchors.topMargin: 10
        Text {
            id: oculusRiftText
            x: leftMargin
            anchors.verticalCenter: parent.verticalCenter
            color: "#ffffff"
            text: qsTr("Oculus Rift")
            font.pixelSize: lblFont
        }

        Switch{
            id: oculusRiftBtn
            width: 50
            height: 30
            checked: true
            anchors.left: oculusRiftText.right
            anchors.leftMargin: leftMargin
        }
    }

    Rectangle {
        id: okRectangle
        width: parent.width * 0.5
        height: 40
        color: "#0e3e64"
        opacity: 1
        z: 1
        anchors {
            left: parent.left
            bottom: parent.bottom
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                toolbox.clearSelected();
                statusItem.setPlayAndPause();
                aboutBox.state = "collapsed";
                setOculus();
            }
        }

        Image {
            id: okImage
            x: (175 - width) / 2
            y: (parent.height - height) / 2
            width: 25
            height: 25
            fillMode: Image.PreserveAspectFit
            source: "../../resources/btn_ok.PNG"
        }
    }

    Rectangle {
        id: cancelRectangle
        width: parent.width * 0.5
        height: 40
        color: "#1f1f1f"
        opacity: 1
        z: 1
        anchors {
            right: parent.right
            bottom: parent.bottom
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                toolbox.clearSelected()
                statusItem.setPlayAndPause();
                aboutBox.state = "collapsed"
            }
        }

        Image {
            id: cancelImage
            x: (175 - width) / 2
            y: (parent.height - height) / 2
            width: 25
            height: 25
            fillMode: Image.PreserveAspectFit
            source: "../../resources/btn_cancel.PNG"
        }


    }

    function setOculus(){
        qmlMainWindow.enableOculus(oculusRiftBtn.checked);
    }

    function getOculus()
    {
        if(qmlMainWindow.getOculus()){
            oculusRiftBtn.checked = true;
        } else{
            oculusRiftBtn.checked = false;
        }
    }

}
