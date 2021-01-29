import QtQuick 2.5
import QtQuick.Layouts 1.1
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick 2.5
import "../../resources"
import "../components"
import "../controls"
import "."

Item {
    id: cameraParameterItem
    width: 80
    height: 50

    property string       title: ""
    property double       maximumValue:  0
    property double       miniumValue:  0
    property double       stepSize: 0
    property double          value


    property int       itemHeight: 30

    property color     hoverColor: "#353535"
    property color     hoverBorderColor : "#4e8d15"

    property int       fontSize: 14
    property color     textColor: "#ffffff"

    property int       margin: 20
    property int       spacing: 20

    signal parameterValueChanged()

    Component.onCompleted: {
        width = parent.width
    }

    Text {
       id: titleText
       x:margin
       y:0
       text: qsTr(title)
       verticalAlignment: Text.AlignVCenter
       color: textColor
       font.pixelSize: fontSize
   }

   Item {
       id: groupItem
       x: 0
       y: titleText.height
       width: parent.width
       height: itemHeight

       Slider {
           id: slider
           width: (parent.width - 2*margin)* (2/3) - 3
           anchors.left: parent.left
           anchors.leftMargin: margin
           updateValueWhileDragging: true
           value: 0
           maximumValue: cameraParameterItem.maximumValue
           minimumValue: cameraParameterItem.miniumValue
           stepSize: cameraParameterItem.stepSize
       }

       FlatSpin {
           id: spin
           anchors.left: slider.right
           anchors.leftMargin: spacing
           maximumValue: slider.maximumValue
           minimumValue: slider.minimumValue
           stepSize: slider.stepSize
           value: slider.value

           onValueChanged: {
               slider.value = value
               cameraParameterItem.value = value
               parameterValueChanged()
           }
       }
   }

   function setInitialValue(initialValue) {
       slider.value = initialValue
   }
}
