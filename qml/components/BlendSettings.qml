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
    width : 330
    height: 350

    property int        spacing: 20
    property int        lblFont: 14
    property int        itemHeight:30
    property bool       isBlend: true

    ScrollView {
        id: scrollView
        width: parent.width
        height: parent.height
        verticalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        flickableItem.interactive: true

        style: ScrollViewStyle {
            transientScrollBars: true
            handle: Item {
                implicitWidth: 14
                implicitHeight: 26
                Rectangle {
                    color: "#424246"
                    anchors.fill: parent
                    anchors.topMargin: 6
                    anchors.leftMargin: 4
                    anchors.rightMargin: 4
                    anchors.bottomMargin: 6
                }

            }

            scrollBarBackground: Item {
                implicitWidth: 14
            }

            decrementControl: Rectangle{
                visible: false
            }

            incrementControl: Rectangle{
                visible: false
            }
            corner: Rectangle{
                visible:false
            }

        }

        Item{
            id: groupItems
            width: scrollView.width
            height: 400

            Item {
                id: radioItem
                width: 330
                height: itemHeight
                anchors.top: parent.top
                anchors.topMargin: spacing

                RowLayout {
                    anchors.left: parent.left
                    anchors.leftMargin: spacing
                    ExclusiveGroup { id: tabPositionGroup }
                    RadioButton {
                        id: featherRadio
                        checked: true
                        exclusiveGroup: tabPositionGroup
                        onClicked: setFeatherMode();
                    }
                    Text {
                        id: feather
                        anchors.left: featherRadio.right
                        anchors.leftMargin: 5
                        width: 70
                        height: parent.height
                        color: "#ffffff"
                        text: "Feather"
                        font.pointSize: 10
                    }

                    RadioButton {
                        id: blendRadio
                        anchors.left: feather.right
                        anchors.leftMargin: spacing
                        exclusiveGroup: tabPositionGroup
                        onClicked: setBlendMode();
                    }

                    Text {
                        id: blendText
                        anchors.left: blendRadio.right
                        anchors.leftMargin: 5
                        width: 70
                        height: parent.height
                        color: "#ffffff"
                        text: "Multi Band"
                        font.pointSize: 10
                    }
                }
            }

            Text {
                   id: leftLabel
                   text: qsTr("Left")
                   verticalAlignment: Text.AlignVCenter
                   color: "#ffffff"
                   font.pixelSize: lblFont
                   anchors.bottom:　leftItem.top
                   anchors.left: parent.left
                   anchors.leftMargin: spacing
               }

               Item {
                   id: leftItem
                   width: 330
                   height: itemHeight
                   anchors.top: radioItem.bottom
                   anchors.topMargin: spacing

                   Slider {
                       id: leftSlider
                       value: leftText.text
                       updateValueWhileDragging: true
                       width: parent.width * 0.85 - spacing * 2
                       minimumValue: 0
                       maximumValue: qmlMainWindow.xRes
                       stepSize: 1
                       anchors.left: parent.left
                       anchors.leftMargin: spacing
                       anchors.verticalCenter: parent.verticalCenter
                       onValueChanged: {
                           if(pressed){
                               if(qmlMainWindow.getLensType() === 0)
                                   rightSlider.value = rightSlider.maximumValue - leftSlider.value;
                               setBlendSettings();
                           }
                       }
                   }

                   FlatText {
                       id: leftText
                       anchors.right: parent.right
                       anchors.rightMargin: spacing
                       width: parent.width * 0.15
                       text: leftSlider.value
                       maximumLength: 6

                       onEditingFinished: {
                           if(qmlMainWindow.getLensType() === 0){
                               leftSlider.value = leftText.text;
                               rightSlider.value = rightSlider.maximumValue - leftText.text;
                           } else  if (qmlMainWindow.getLensType() === 1){
                               leftSlider.value = leftText.text;
                           }

                           setBlendSettings();
                       }
                   }
               }

               Text {
                   id: rightLabel
                   text: qsTr("Right")
                   verticalAlignment: Text.AlignVCenter
                   color: "#ffffff"
                   font.pixelSize: lblFont
                   anchors.bottom:　rightItem.top
                   anchors.left: parent.left
                   anchors.leftMargin: spacing
               }


               Item {
                   id: rightItem
                   width: 330
                   height: itemHeight
                   anchors.top: leftItem.bottom
                   anchors.topMargin: topMargin

                   Slider {
                       id: rightSlider
                       value: rightText.text
                       minimumValue: 0
                       maximumValue: qmlMainWindow.xRes
                       stepSize: 1
                       anchors.left: parent.left
                       anchors.leftMargin: spacing
                       anchors.verticalCenter: parent.verticalCenter
                       width: parent.width * 0.85 - spacing * 2
                       activeFocusOnPress: true
                       updateValueWhileDragging: true

                       onValueChanged: {
                           if(pressed){
                               if(qmlMainWindow.getLensType() === 0)
                                   leftSlider.value = leftSlider.maximumValue - rightSlider.value;
                               setBlendSettings();
                           }

                       }

                   }

                   FlatText {
                       id: rightText
                       anchors.right: parent.right
                       anchors.rightMargin: spacing
                       width: parent.width * 0.15
                       text: rightSlider.value
                       maximumLength: 6

                       onEditingFinished: {
                           if(qmlMainWindow.getLensType() === 0){
                               leftSlider.value = leftSlider.maximumValue - rightText.text;
                               rightSlider.value = rightText.text;
                           } else if(qmlMainWindow.getLensType() === 1){
                               rightSlider.value = rightText.text;                               
                           }
						   setBlendSettings();
                       }
                   }
               }

               Text {
                   id: topLabel
                   text: qsTr("Top")
                   verticalAlignment: Text.AlignVCenter
                   color: "#ffffff"
                   font.pixelSize: lblFont
                   anchors.bottom:　topItem.top
                   anchors.left: parent.left
                   anchors.leftMargin: spacing
               }

               Item {
                   id: topItem
                   width: 330
                   height: itemHeight
                   anchors.top: rightItem.bottom
                   anchors.topMargin: spacing

                   Slider {
                       id: topSlider
                       value: topText.text
                       minimumValue: 0
                       maximumValue: qmlMainWindow.yRes
                       stepSize: 1
                       anchors.left: parent.left
                       anchors.leftMargin: spacing
                       anchors.verticalCenter: parent.verticalCenter
                       width: parent.width * 0.85 - spacing * 2
                       updateValueWhileDragging: true
                       onValueChanged: {
                           if(pressed){
                               if(qmlMainWindow.getLensType() === 0)
                                   bottomSlider.value = bottomSlider.maximumValue - topSlider.value;
                               setBlendSettings();
                           }
                       }
                   }

                   FlatText {
                       id: topText
                       anchors.right: parent.right
                       anchors.rightMargin: spacing
                       width: parent.width * 0.15
                       text: topSlider.value
                       maximumLength: 6

                       onEditingFinished: {
                           if(qmlMainWindow.getLensType() === 0)
                           {
                               topSlider.value = topText.text;
                               bottomSlider.value = bottomSlider.maximumValue - topText.text;
                           } else if (qmlMainWindow.getLensType() === 1){
                               topSlider.value = topText.text;                               
                           }
						   setBlendSettings();
                       }
                   }
               }

               Text {
                   id: bottomLabel
                   text: qsTr("Bottom")
                   verticalAlignment: Text.AlignVCenter
                   color: "#ffffff"
                   font.pixelSize: lblFont
                   anchors.bottom:　bottomItem.top
                   anchors.left: parent.left
                   anchors.leftMargin: spacing
               }

               Item {
                   id: bottomItem
                   width: 330
                   height: itemHeight
                   anchors.top: topItem.bottom
                   anchors.topMargin: topMargin

                   Slider {
                       id: bottomSlider
                       value: bottomText.text
                       minimumValue: 0
                       maximumValue: qmlMainWindow.yRes
                       stepSize: 1
                       anchors.left: parent.left
                       anchors.leftMargin: spacing
                       anchors.verticalCenter: parent.verticalCenter
                       width: parent.width * 0.85 - spacing * 2
                       updateValueWhileDragging: true
                       onValueChanged: {
                           if(pressed){
                               if(qmlMainWindow.getLensType() === 0)
                                   topSlider.value = topSlider.maximumValue - bottomSlider.value;
                               setBlendSettings();
                           }
                       }
                   }

                   FlatText {
                       id: bottomText
                       anchors.right: parent.right
                       anchors.rightMargin: spacing
                       width: parent.width * 0.15
                       text: bottomSlider.value
                       maximumLength: 6

                       onEditingFinished: {
                           if(qmlMainWindow.getLensType() === 0){
                               topSlider.value = topSlider.maximumValue - bottomText.text;
                               bottomSlider.value = bottomText.text;
                           } else if(qmlMainWindow.getLensType() === 1){
                               bottomSlider.value = bottomText.text;                               
                           }
						   setBlendSettings();
                       }
                   }

               }

            Item {
                id: levelItem
                width: 330
                height: itemHeight
                anchors.top: bottomItem.bottom
                anchors.topMargin: topMargin
                visible: true

                Text {
                    id: levelLabel
                    text: qsTr("Max Level")
                    verticalAlignment: Text.AlignVCenter
                    color: "#ffffff"
                    font.pixelSize: lblFont
                    anchors.bottom: levelItem.top
                    anchors.left:　parent.left
                    anchors.leftMargin: spacing
                }
                Slider {
                    id: levelSlider
                    value: levelText.text
                    maximumValue: 5
                    minimumValue: 0
                    stepSize: 1
                    anchors.left: parent.left
                    anchors.leftMargin: spacing
                    anchors.verticalCenter: parent.verticalCenter
                    width: parent.width * 0.85 - spacing * 2
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if(pressed){
                            setBlendSettings();
                        }
                    }
                }

                FlatText {
                    id: levelText
                    anchors.right: parent.right
                    anchors.rightMargin: spacing
                    width: parent.width * 0.15
                    text: levelSlider.value
                    maximumLength: 6
                    onEditingFinished: {
                        levelSlider.value = levelText.text;
                        setBlendSettings();
                    }
                }
            }
        }
    }

    function getBlendSettings()
    {
        leftSlider.maximumValue = qmlMainWindow.xRes;
        rightSlider.maximumValue = qmlMainWindow.xRes;
        topSlider.maximumValue = qmlMainWindow.yRes;
        bottomSlider.maximumValue = qmlMainWindow.yRes;
        leftSlider.value = qmlMainWindow.getLeft(cameraCombo.currentIndex);
        rightSlider.value = qmlMainWindow.getRight(cameraCombo.currentIndex);
        topSlider.value = qmlMainWindow.getTop(cameraCombo.currentIndex);
        bottomSlider.value = qmlMainWindow.getBottom(cameraCombo.currentIndex);
        getBlendMode();
    }

    function setBlendSettings(){

        if(!qmlMainWindow.start) return;

        qmlMainWindow.setLeft(leftSlider.value, cameraCombo.currentIndex);
        qmlMainWindow.setRight(rightSlider.value, cameraCombo.currentIndex);
        qmlMainWindow.setTop(topSlider.value, cameraCombo.currentIndex)
        qmlMainWindow.setBottom(bottomSlider.value, cameraCombo.currentIndex);

        if(isBlend){
            qmlMainWindow.setBlendMode(2);
            qmlMainWindow.setBlendLevel(levelSlider.value);
        }else{
            qmlMainWindow.setBlendMode(1);
        }
        qmlMainWindow.reStitch(true);

    }

    function getBlendMode()
    {
        var blendMode = qmlMainWindow.getBlendMode();

        if(blendMode === 1){
            featherRadio.checked = true;
            blendRadio.checked = false;
            levelItem.visible = false;

			isBlend = false;
        } else if(blendMode === 2){
            featherRadio.checked = false;
            blendRadio.checked = true;
            levelItem.visible = true;
            levelSlider.value = qmlMainWindow.getBlendLevel();

			isBlend = true;
        }		
    }

    function setFeatherMode(){
        levelItem.visible = false;
        isBlend = false;
        setBlendSettings();
    }

    function setBlendMode(){
        levelItem.visible = true;
        isBlend = true;
        setBlendSettings();
    }

}
