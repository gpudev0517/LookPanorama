import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import "../controls"

ToolWindow{
    id: details
    width: 350
    height: 100
    z: 10
    windowState: qsTr("windowed")
    visible: true

    property string   title: "Color Temperature"
    property int      fontSize: 15
    property int      itemHeight: 30
    property int      spacing: 20
    property color    textColor: "#ffffff"
    property var      currentCursorShape: Qt.CrossCursor
    property int      lblWidth: 60

	property int        leftMargin: 20
	property int        rightMargin: 0	
	property color      spliterColor: "#555555"
	property int        lblFont: 14		
	property var        opacityVal: 0.9	

    property int        minTemperature: 2000
    property int        maxTemperature: 12000
    property int        defaultTemperature: 6600

	Item {
        id: ctSettingboxItem
		width : 330
        height: parent.height

       Slider {
           id: leftSlider
           value: defaultTemperature
           updateValueWhileDragging: true
           width: parent.width * 0.85 - spacing * 2
           minimumValue: minTemperature
           maximumValue: maxTemperature
           stepSize: 1
           anchors.top: parent.top
           anchors.topMargin: spacing
           anchors.left: parent.left
           anchors.leftMargin: spacing
           onValueChanged: {
               leftText.text = value;
               qmlMainWindow.setColorTemperature(value);
           }
       }

       FlatText {
           id: leftText
           anchors.top: parent.top
           anchors.topMargin: spacing
           anchors.right: parent.right
           anchors.rightMargin: rightMargin
           width: parent.width * 0.15
           text: leftSlider.value
           maximumLength: 6

           onEditingFinished: {
               leftSlider.value = leftText.text;			   
           }
       }
   }	
   
   function setColorTemperature(ctValue) {
		leftSlider.value = ctValue;		
   }   
}
