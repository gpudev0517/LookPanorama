import QtQuick 2.0
import QtQuick.Dialogs 1.2
import QtQuick.Controls 1.4

Item {
    width: 500
    height: 48
    property bool isHoveredCheck: false
    property bool isHoveredUncheck: false
    property bool isHoveredMore: false
    property bool isHover: false

    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#1f1f1f"
    }

    ToolbarItem {
                id: bannerCtrl
                anchors.right: spliter.left
                imgUrl: "../../resources/ico_banner.png"
                title: ""
                theme: applicationSettingWindow.uiTheme
                autoHide: root.isHiddenToolbarItem
                fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"

                onClicked: {

                }
            }

    Item {
        id: spliter
        y: (parent.height - height) / 2
        width: 2
        height: parent.height - 20
        anchors.right: bannerListItem.left
        anchors.rightMargin: 13
        Rectangle{
            color: "#1f1f1f"
            x: 1
            height: parent.height
            width: 1

        }
        Rectangle{
            color: "#3f3f3f"
            x: 2
            height: parent.height
            width: 1
        }
    }

    Item{
        id: bannerListItem
        anchors.right: deleteItem.left
        anchors.rightMargin: 4
        width: 170
        height: 48
        visible: true

        Text {
            id: bannerLabel
            anchors.left: parent.left
            anchors.leftMargin: 5
            anchors.verticalCenter: parent.verticalCenter
            text: qsTr("Banners")
            color: "#ffffff"
            font.pixelSize: 14

        }

        ComboBox {
            id: bannerCombo
            anchors.left: bannerLabel.right
            anchors.leftMargin: 17
            anchors.verticalCenter: parent.verticalCenter
            width: 100
            height: 28
            model: ListModel {
                id: bannerlistModel
            }

            onCurrentTextChanged:
            {
            }
        }
    }

    ToolbarItem {
                id: deleteItem
                anchors.right: checkItem.left
                imgUrl: "../../resources/icon_delete.png"
                title: ""
                theme: applicationSettingWindow.uiTheme
                autoHide: root.isHiddenToolbarItem
                fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"

                onClicked: {
                    if(sphericalView.bannerIndex == -1) return;

                    var deleteBannerIdx = bannerCombo.currentIndex;
                    qmlMainWindow.removeBannerAtIndex(deleteBannerIdx);
                    removeBanneListAtIndex(deleteBannerIdx);

                    updateAllBannerList();
                    if(bannerlistModel.count == 0)
                        initBannerList();
                }
            }

    ToolbarItem {
                id: checkItem
                anchors.right: parent.left
                imgUrl: "../../resources/check.png"
                title: ""
                theme: applicationSettingWindow.uiTheme
                autoHide: root.isHiddenToolbarItem
                fontURL: applicationSettingWindow.uiTheme == "Default"? "":"../../resources/font/MTF Base Outline.ttf"

                onClicked: clearBanner();
            }

    function clearBanner(){
        toolbox.clearSelected();
        statusItem.setPlayAndPause();
        toolbox.initSphericalTopControls();
        sphericalView.isBanner = false;
        sphericalView.clearAllBannerPoint();
    }

    function insertBannerList(bannerIndex){
        bannerCombo.enabled = true;
        bannerCombo.currentIndex = bannerIndex;
        bannerlistModel.append({"text": "Banner" + (bannerIndex + 1)})
     }

    function updateBannerList(index){
        bannerlistModel.set(index,{"text": "Banner" + (index + 1)});
    }

    function updateAllBannerList(){
        bannerlistModel.clear();
        for(var i = 0; i < sphericalView.bannerIndex + 1; i++){
            updateBannerList(i);
        }
    }

    function removeBanneListAtIndex(index){
        bannerlistModel.remove((index),1);
        bannerCombo.currentIndex = 0;
        sphericalView.bannerIndex--;
    }

    function removeLastBannerList(){
        var lastIndex = bannerlistModel.count - 1;
        removeBanneListAtIndex(lastIndex);
    }

    function initBannerList(){
        bannerCombo.enabled = false;
        bannerCombo.currentIndex = -1;
    }

    function bannerListCount(){
        return bannerlistModel.count;
    }

    function clearAllBannerList(){
        bannerlistModel.clear();
        var bannerCount = qmlMainWindow.getBannerCount();
        if (bannerCount > 0){
            sphericalView.bannerIndex = bannerCount - 1;
            for (var i = 0 ; i < bannerCount; i ++)
                insertBannerList(i);
        } else {
            sphericalView.bannerIndex = -1;
            bannerCombo.enabled = false;
            bannerCombo.currentIndex = -1;

        }

    }
}

