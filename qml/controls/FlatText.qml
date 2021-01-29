import QtQuick 2.0
import QtQuick.Controls.Styles 1.4
import QtQuick.Controls 1.4

TextField {
    width:100
    height: 30
    text: qsTr("")
    font.pixelSize: 12

   style: TextFieldStyle {
           textColor: "#4e8d15"
           background: Rectangle {
               radius: 2
               implicitWidth: 100
               implicitHeight: 24
               border.color: "#4e8d15"
               border.width:1
               color:"#00000000"
           }
       }

	onEditingFinished: {               
		root.setFocus();
    }

    function setText(str) {
        text = str
    }
}

