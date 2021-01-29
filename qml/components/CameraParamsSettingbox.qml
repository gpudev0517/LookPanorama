import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import "../controls"
import "../components"

ToolWindow{
    id: cameraParamsSettingbox
    width: 350
    height: 800
    z: 10
    windowState: qsTr("windowed")
    visible: true

    property string   title: "Camera Parameters"
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
	property bool       isCPoint: false
	property int        cameraCnt
	property var        opacityVal: 0.9
	property int        camWidth: 100
	property int        camHeight: 100    
    property int        m_curCameraIndex;
    property bool       m_isReady: false

  	Item {
        id: cameraParamsItem
		width : 330
        height: parent.height

		property bool       isHovered : false		

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
				
				anchors.topMargin: spacing
				width: scrollView.width
				height: parent.height

				function convertAngle(angleText)
				{
					var val = Number(angleText);
					while (val > 180)
						val = val - 360;
					while(val < -180)
						val = val + 360;
					return val;
				}

                Item {
                    id: lensTypeItem
                    width: parent.width
                    height: 30
                    anchors.top: parent.top
                    anchors.topMargin: spacing

                    Text {
                        id: lensTypeText
                        color: textColor
                        width: lblWidth
                        text: qsTr("Lens Type")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: spacing
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: 13
                    }

                    ComboBox {
                        id: lensTypeCombo
                        width: parent.width / 2
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: lensTypeText.right
                        anchors.leftMargin: spacing
                        model: ["Standard", "Fullframe Fisheye", "Circular Fisheye", "CV Standard", "CV Fisheye"]
						enabled: false
                        onCurrentTextChanged: {
                            qmlMainWindow.setLensType(currentIndex);
                            getCameraParams(m_curCameraIndex);
                        }
                    }
                }

                CameraParameterItem {
                    id: leftItem
                    title: qsTr("Left")
                    anchors.top: lensTypeItem.bottom

                    miniumValue: 0
                    maximumValue: qmlMainWindow.xRes

                    stepSize: 1

                    onParameterValueChanged: {
                        if(qmlMainWindow.getLensType() === 0 || qmlMainWindow.getLensType() === 3)
                            rightItem.initializeValue = rightItem.maximumValue - value;
                        cameraParamsItem.setBlendSettings();
                    }
                }

                CameraParameterItem {
                    id: rightItem
                    title: qsTr("Right")
                    anchors.top: leftItem.bottom

                    miniumValue: 0
                    maximumValue: qmlMainWindow.xRes

                    stepSize: 1

                    onParameterValueChanged: {
                        if(qmlMainWindow.getLensType() === 0 || qmlMainWindow.getLensType() === 3)
                            leftItem.initializeValue = leftItem.maximumValue - value;
                        cameraParamsItem.setBlendSettings();
                    }
                }

                CameraParameterItem {
                    id: topItem
                    title: qsTr("Top")
                    anchors.top: rightItem.bottom

                    miniumValue: 0
                    maximumValue: qmlMainWindow.yRes

                    stepSize: 1

                    onParameterValueChanged: {
                        if(qmlMainWindow.getLensType() === 0 || qmlMainWindow.getLensType() === 3)
                            bottomItem.initializeValue = bottomItem.maximumValue - value;
                        cameraParamsItem.setBlendSettings();
                    }
                }

                CameraParameterItem {
                    id: bottomItem
                    title: qsTr("Bottom")
                    anchors.top: topItem.bottom

                    miniumValue: 0
                    maximumValue: qmlMainWindow.yRes

                    stepSize: 1

                    onParameterValueChanged: {
                        if(qmlMainWindow.getLensType() === 0 || qmlMainWindow.getLensType() === 3)
                            topItem.initializeValue = topItem.maximumValue - value;
                        cameraParamsItem.setBlendSettings();
                    }
                }

                CameraParameterItem {
                    id: yawItem
                    title: qsTr("Yaw")
                    anchors.top: bottomItem.bottom

                    miniumValue: -180
                    maximumValue: 180

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: pitchItem
                    title: qsTr("Pitch")
                    anchors.top: yawItem.bottom

                    miniumValue: -180
                    maximumValue: 180

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: rollItem
                    title: qsTr("Roll")
                    anchors.top: pitchItem.bottom

                    miniumValue: -180
                    maximumValue: 180

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: fovItem
                    title: qsTr("FoV")
                    anchors.top: rollItem.bottom

                    miniumValue: 0.1
                    maximumValue: 360

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: fovYItem
                    title: qsTr("FocalY")
                    anchors.top: fovItem.bottom

                    miniumValue: 0.1
                    maximumValue: 360

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: k1Item
                    title: qsTr("K1")
                    anchors.top: fovYItem.bottom

                    miniumValue: -2
                    maximumValue: 2

                    stepSize: 0.01

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: k2Item
                    title: qsTr("K2")
                    anchors.top: k1Item.bottom

                    miniumValue: -2
                    maximumValue: 2

                    stepSize: 0.001

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: k3Item
                    title: qsTr("K3")
                    anchors.top: k2Item.bottom

                    miniumValue: -2
                    maximumValue: 2

                    stepSize: 0.0001

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: offsetXItem
                    title: qsTr("OffsetX")
                    anchors.top: k3Item.bottom

                    miniumValue:  -camWidth / 2
                    maximumValue: camWidth / 2

                    stepSize: 0.1

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }

                CameraParameterItem {
                    id: offsetYItem
                    title: qsTr("OffsetX")
                    anchors.top: offsetXItem.bottom

                    miniumValue:  -camWidth / 2
                    maximumValue: camWidth / 2

                    stepSize: 0.1

                    onParameterValueChanged: {
                        cameraParamsItem.setCameraParams();
                    }
                }
            }
		}

        function getCameraParams(curCameraIndex){

            if(!qmlMainWindow.start) return;

            m_isReady = false;

            camWidth = qmlMainWindow.getWidth(curCameraIndex);
            camHeight = qmlMainWindow.getHeight(curCameraIndex)

            yawItem.setInitialValue(qmlMainWindow.getYaw(curCameraIndex))
            pitchItem.setInitialValue(qmlMainWindow.getPitch(curCameraIndex))
            rollItem.setInitialValue(qmlMainWindow.getRoll(curCameraIndex))
            fovItem.setInitialValue(qmlMainWindow.getFov(curCameraIndex))
            fovYItem.setInitialValue(qmlMainWindow.getFovy(curCameraIndex))
            k1Item.setInitialValue(qmlMainWindow.getK1(curCameraIndex))
            k2Item.setInitialValue(qmlMainWindow.getK2(curCameraIndex))
            k3Item.setInitialValue(qmlMainWindow.getK3(curCameraIndex))
            offsetXItem.setInitialValue(qmlMainWindow.getOffsetX(curCameraIndex))
            offsetYItem.setInitialValue(qmlMainWindow.getOffsetY(curCameraIndex))



			// check lensType
            var lensType = qmlMainWindow.getLensType();
            lensTypeCombo.currentIndex = lensType;
            if (lensType === 3) {
				// LensType_opencvLens_Standard
                fovItem.title = "FoV";
                fovItem.visible = false;
				fovYItem.visible = false;						 
				k1Item.anchors.top = fovItem.bottom;
			} else if (lensType == 4) {
				// LensType_opencvLens_Fisheye
                fovItem.title = "FoVX";
                fovYItem.title = "FoVY";
                fovYItem.visible = true;
				fovYItem.visible = true;			
				k1Item.anchors.top = fovYItem.bottom;
			} else {
                fovItem.title = "FoV";
                fovItem.maximumValue = 360;
                fovYItem.visible = false;
				fovYItem.visible = false;			 
				k1Item.anchors.top = fovItem.bottom;
			}

            // Get blend setting values
            leftItem.setInitialValue(qmlMainWindow.getLeft(curCameraIndex))
            rightItem.setInitialValue(qmlMainWindow.getRight(curCameraIndex))
            topItem.setInitialValue(qmlMainWindow.getTop(curCameraIndex))
            bottomItem.setInitialValue(qmlMainWindow.getBottom(curCameraIndex))

            m_isReady = true;
        }

        function setCameraParams(){

            if(!qmlMainWindow.start || !m_isReady) return;

            qmlMainWindow.setYaw(yawItem.value, m_curCameraIndex);
            qmlMainWindow.setPitch(pitchItem.value, m_curCameraIndex);
            qmlMainWindow.setRoll(rollItem.value, m_curCameraIndex);
            qmlMainWindow.setFov(fovItem.value, m_curCameraIndex);
            qmlMainWindow.setFovy(fovYItem.value, m_curCameraIndex);
            qmlMainWindow.setK1(k1Item.value, m_curCameraIndex);
            qmlMainWindow.setK2(k2Item.value, m_curCameraIndex);
            qmlMainWindow.setK3(k3Item.value, m_curCameraIndex);
            qmlMainWindow.setOffsetX(offsetXItem.value, m_curCameraIndex);
            qmlMainWindow.setOffsetY(offsetYItem.value, m_curCameraIndex);
            qmlMainWindow.reStitch(true);
			sphericalView.updateSeam();
        }

        function insertCamName(){
            cameraModel.clear();
            for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
                cameraModel.append({"text": "Camera" + (i + 1)})
            }
        }        

        function setBlendSettings(){

            if(!qmlMainWindow.start || !m_isReady) return;
			
            qmlMainWindow.setLeft(leftItem.value, m_curCameraIndex);
            qmlMainWindow.setRight(rightItem.value, m_curCameraIndex);
            qmlMainWindow.setTop(topItem.value, m_curCameraIndex)
            qmlMainWindow.setBottom(bottomItem.value, m_curCameraIndex);
						
            qmlMainWindow.reStitch(true);
        }
   }

    function getCameraParams(curCameraIndex){
        m_curCameraIndex = curCameraIndex;
        cameraParamsItem.getCameraParams(curCameraIndex);
    }

    function setCameraParams(){
        cameraParamsItem.setCameraParams();
    }
}
