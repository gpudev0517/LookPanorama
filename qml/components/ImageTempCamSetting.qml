import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2
import "../controls"

Item {
    id: item1
    width : 300
    height: 140
    opacity: 1

    property int        leftMargin: 20
    property int        rightMargin: 20
    property int        spacing: 20
    property color      textColor: "#ffffff"
    property int        lblFont: 14
    property int        itemHeight: 30
    property int        curIndex
    property string     cameraName
    property string     title
    property int        lblWidth: 40

    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#171717"
        border.color: "#1f1f1f"
        border.width: 1
        opacity: 0.9

    }

    ScrollView {
        id: scrollView
        width: parent.width
        height: parent.height
        opacity: 0.8
        verticalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        flickableItem.interactive: true

        style: ScrollViewStyle {
            transientScrollBars: false
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
            x: 0
            y: 0
            width: scrollView.width
            height: scrollView.height

            Item {
                id: titleItem
                width: parent.width
                height: 30
                anchors.top: parent.top
                Text {
                    id:  title
                    x: (parent.width - width) / 2
                    anchors.verticalCenter: parent.verticalCenter
                    color: "#ffffff"
                    text: qsTr("")
                    font.pixelSize: 15
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }

            Spliter {
                width: parent.width - 4
                x: ( parent.width - width ) / 2
                anchors.top: titleItem.bottom
            }

            Item {
                id: prefixItem
                width: parent.width
                height: itemHeight
                anchors.topMargin: spacing
                anchors.top: titleItem.bottom

                Text {
                    id: prefixLabel
                    width: lblWidth
                    color: textColor
                    text: qsTr("Prefix")
                    horizontalAlignment: Text.AlignLeft
                    font.pixelSize: lblFont
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                }

                FlatText {
                    id: filePrefix
                    anchors.right: parent.right
                    anchors.rightMargin: spacing
                    width: parent.width * 0.7
                    height: itemHeight
                    anchors.verticalCenter: parent.verticalCenter
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
                        var prefix = filePrefix.text;
                        qmlMainWindow.setTempImagePrefix(curIndex,prefix);
                        imageList.set(curIndex,{"prefixText": prefix});
                        imageTempCamSetting.state = "collapsed";
                    }
                }

                Image {
                    id: okImage
                    x: (parent.width - width) / 2
                    y: (parent.height - height) / 2
                    width: 25
                    height: 25
                    fillMode: Image.PreserveAspectFit
                    source: "../../resources/check.png"
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
                        imageTempCamSetting.state = "collapsed";
                    }
                }

                Image {
                    id: cancelImage
                    x: (parent.width - width) / 2
                    y: (parent.height - height) / 2
                    width: 25
                    height: 25
                    fillMode: Image.PreserveAspectFit
                    source: "../../resources/uncheck.png"
                }
            }
         }
     }



    function getCameraSetting(imageIndex){
        curIndex = imageIndex;
        filePrefix.text = qmlMainWindow.getTempImagePrefix(curIndex);
    }

    function setCameraValues(imageIndex)
    {
        curIndex = imageIndex;
        var item = imageList.get(curIndex);
        cameraName = item.titleText;
        var pos = cameraName.lastIndexOf("/") + 1;
        title.text = cameraName.substring(pos);

    }

    function setFileExt()
    {
        switch (fileExtCombo.currentIndex)
        {
            case 0: qmlMainWindow.setTempImageExt(curIndex,"dpx"); break
            case 1: qmlMainWindow.setTempImageExt(curIndex,"tiff"); break;
            case 2: qmlMainWindow.setTempImageExt(curIndex,"jpg"); break;
            case 3: qmlMainWindow.setTempImageExt(curIndex,"png"); break;
        }
    }

    function getFileExt()
    {
        var fileExt = qmlMainWindow.getTempImageExt(curIndex);
        if (fileExt === "dpx")
        {
            fileExtCombo.currentIndex = 0;
        }
        else if (fileExt === "tiff")
        {
            fileExtCombo.currentIndex = 1;
        }
        else if (fileExt === "jpg")
        {
            fileExtCombo.currentIndex = 2;
        }
        else if (fileExt === "png")
        {
            fileExtCombo.currentIndex = 3
        }
        else if (fileExt === "")
        {
            fileExtCombo.currentIndex = 0;
        }
    }

}
