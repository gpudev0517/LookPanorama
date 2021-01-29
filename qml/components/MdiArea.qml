import QtQuick 2.4
import QtQuick.Window 2.2

Rectangle {
    MouseArea {
        anchors.fill: parent
        id: ma
        onClicked: parent.clicked(mouse)
        onDoubleClicked: parent.doubleClicked(mouse)
    }

    property var windowMenuBarMenu
    property var activeWindow
    property var topWindowGroups
    property alias cursorShape: ma.cursorShape
    signal clicked(var mouse)
    signal doubleClicked(var mouse)
    property bool somethingIsDragged: false
//    property bool autoPositionGroups: []



    Component.onCompleted: {
        for(var i in children) {
           if(!!children[i].floatingWindow) {
               if(!(children[i].windowMenuGroup in topWindowGroups))
                   console.log('MainArea: ' + ' Window "' + children[i].caption + '" at ' + i + ' is not registered in topWindowGroups!')
           }
        }

    }
    function getWindowAt(x, y) {
        var c = childAt(x, y)
        var p = c
        while(p.parent != this) p = p.parent;
//        console.log(p)
//        console.log(p.x, p.y, p.width, p.height)
        return p
    }
}
