import QtQuick 2.4
import QtQuick.Window 2.2
Item {
    id: root

    onChildrenChanged: {
//        children[0].width = Qt.binding(function())
    }
    Binding {
        when: children.length > 0
        target: children[0]
        property: "width"
        value: root.width * Screen.devicePixelRatio
    }
    Binding {
        when: children.length > 0
        target: children[0]
        property: "height"
        value: root.height * Screen.devicePixelRatio
    }
    Binding {
        when: children.length > 0
        target: children[0]
        property: "transform"
        value: Scale {
            xScale: 1.0 / Screen.devicePixelRatio
            yScale: 1.0 / Screen.devicePixelRatio
        }
    }
}

