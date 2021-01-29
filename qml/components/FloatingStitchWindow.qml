import QtQuick 2.4
import QtQuick.Window 2.2
Item {
    property bool __floatingWindow__: true
    default property alias content: area.children
    property alias handle: content.handle
    property var caption: ""
    property bool hidden: false
    property bool floating: false
    property bool resizable: false
    property string windowMenuGroup
    property var menu
    property var __menuItem__
    property int minimumWidth: 0
    property int minimumHeight: 0
    property double maximumWidth: 5000
    property double maximumHeight: 5000
    property int mdiStickyDistance: 10
    property var floatingWindowBorderColor: "#4e8d15"
    property int animationDelays: 200
    property var floatingWindowBackgroundColor: "#1f1f1f"
    property real minimizedWidth: width
    property real prevX: x
    property real prevY: y
    property real prevHeight : height
    property real prevWidth: width
    property bool windowReady: false
    //onWindowReadyChanged: {parent.setDefaultToolwindowPositionsDelayed()}

    id: root
    width: 300
    height: 300


    onWidthChanged: {
        if(!animationsActive&&width < minimumWidth && windowState != "minimized") width = minimumWidth
        if(!animationsActive&&width > maximumWidth && windowState != "minimized") width = maximumWidth
        if(windowState == "windowed") prevWidth = width
    }

    onHeightChanged:{
        if(!animationsActive && height < minimumHeight&& windowState != "minimized") height = minimumHeight
        if(!animationsActive && height > maximumHeight&& windowState != "minimized") height = maximumHeight
        if(content.handle != undefined &&
                height < content.handle.height + 2*content.border.width) height = content.handle.height + 2*content.border.width
        if(windowState == "windowed") prevHeight = height

    }
    onXChanged: if(windowState == "windowed" ) prevX = x
    onYChanged: if(windowState == "windowed") prevY = y

    property string windowState: "windowed"
    onWindowStateChanged:  {
        if(windowState != "windowed" && windowState != "minimized" && windowState != "maximized")
            windowState = "windowed";
    }

    visible: !floating && !hidden
    states: [
        State {
            name: "maximized"
            when: windowState == "maximized"
            StateChangeScript {
                name: "myScript"
                script: {
                    /*if(floating && window.visibility != Window.Maximized)
                        window.showMaximized()*/
                    if(!floating)
                        handleZ();
                }
            }
            AnchorChanges {
                target: root
                anchors.left: !floating?parent.left:undefined
                anchors.right: !floating?parent.right:undefined
                anchors.top: !floating?parent.top:undefined
                anchors.bottom: !floating?parent.bottom:undefined
            }

        },
        State {
            name: "minimized"
            when: windowState == "minimized"
            StateChangeScript {
                name: "myScript"
                script: {
                    if(floating && window.visibility != Window.Minimized)
                        window.showMinimized();
                    prevX = x
                    x = prevX + width - minimizedWidth
                }
            }
            PropertyChanges {
                target: root
                height: !floating?handle.height + 2:prevHeight


                width:minimizedWidth
                y:prevY
                explicit: true
            }

        },
        State {
            name: "windowed"
            when: windowState == "windowed"
            StateChangeScript {
                name: "myScript"
                script: {
                    if(floating && window.visibility != Window.Windowed)
                        window.showNormal();
                    windowReady = true
                }
            }
            AnchorChanges {
                target: root
                anchors.left: undefined
                anchors.right: undefined
                anchors.top: undefined
                anchors.bottom: undefined
            }

            PropertyChanges {
                target: root
                height: prevHeight
                width:prevWidth
                x:prevX
                y:prevY
                explicit: true

            }
        }
    ]


    transitions: [
        Transition {
            from: "*"
            to: "*"
            NumberAnimation{properties: "x,y,height,width"; duration: animationDelays; easing.type: Easing.InOutCubic }
            AnchorAnimation { duration: animationDelays; easing.type: Easing.InOutCubic }
        }
    ]

    Component.onCompleted: {
      //  if (root.parent.topWindowGroups[windowMenuGroup] != undefined)
        //    root.z += root.parent.topWindowGroups[windowMenuGroup].z_mult;
        state = "maximized"

        prevX = x;
        prevY = y;
        prevHeight = height;
        prevWidth = width;

        content.checkHandle();

        //        if(width < minimumWidth)
        //            width = minimumWidth;

        //        if(height < minimumHeight)
        //            height = minimumHeight;

        windowReady = true
    }



    onHiddenChanged: {
        if(__menuItem__==undefined) return;
        __menuItem__.checked = !hidden
    }

    onMenuChanged: {
        if(menu == undefined) {
            return;
        }
        if(typeof(menu.insertItem) != 'function')
            return;
        if(__menuItem__ != undefined) destroy(__menuItem__);
        //TODO: window menu groups
        __menuItem__ = menu.insertItem(2, '');


        //__menuItem__.text = Qt.binding(function(){return caption;});
        __menuItem__.checkable = true;
        //__menuItem__.checked = !hidden;
        __menuItem__.toggled.connect(function(checked){
            hidden = !checked;
            //            if(!checked)
            root.parent.setVideoWindowPositions()
        });

    }

    function close() {
        closing();
        menu.removeItem(__menuItem__);
        root.destroy();
    }

    function handleZ() {
//        if(!floating) {
//            for(var i = 0; i < root.parent.children.length; i ++) {
//                if(!!root.parent.children[i].__floatingWindow__) {
//                    root.parent.children[i].z += root.parent.topWindowGroups[root.parent.children[i].windowMenuGroup].z_mult
//                }
//            }
//            if(root.parent.topWindowGroups[windowMenuGroup].wnd == undefined)

//                root.z += 0.5;

//            else
//                root.z = root.parent.topWindowGroups[windowMenuGroup].wnd.z + 0.5
//            root.parent.topWindowGroups[windowMenuGroup].wnd = root
//        }
    }

    Rectangle {
        MouseArea {
            id: resizeMa
            property var clickPos: "1,1"
            property real movedX: 0
            property real movedY: 0
            property int resizeProcState: -1
            property int cursorState: getState(mouseX, mouseY)
            cursorShape: resizeProcState == -1 ? getCursor(cursorState) : getCursor(resizeProcState)

            anchors.fill: parent
            anchors.margins: -2
            hoverEnabled: resizable
            function getState(x, y) {
                var m = 10
                var resizeType = -1
                //Floating window cannot be resized this way with simultaneous changing of size and position, so disabled.
                if(x < m && y < m && !floating)                      {resizeType = 0;}
                else if(x > width - m && y > height - m)             {resizeType = 1;}
                else if(x > width - m && y < m && !floating)         {resizeType = 2;}
                else if(x < m && y > height - m && !floating)        {resizeType = 3;}
                else if(x < m && !floating)                          {resizeType = 4;}
                else if(x > width - m)                               {resizeType = 5;}
                else if(y < m && !floating)                          {resizeType = 6;}
                else if(y > height - m)                              {resizeType = 7;}
                return resizeType;
            }
            function getCursor(state) {
                if(!resizable) return Qt.ArrowCursor;
                switch(state) {
                case 0:
                case 1:
                    return Qt.SizeFDiagCursor;
                case 2:
                case 3:
                    return Qt.SizeBDiagCursor;
                case 4:
                case 5:
                    return Qt.SizeHorCursor;
                case 6:
                case 7:
                    return Qt.SizeVerCursor;
                default:
                    return Qt.ArrowCursor;
                }
            }

            onPressed: {
                clickPos  = mapToItem(root.parent, mouse.x,mouse.y)
                resizeProcState = getState(mouse.x,mouse.y)
                movedX = 0
                movedY = 0
            }
            onReleased: {
                resizeProcState = -1
                movedX = 0
                movedY = 0
            }
        }

        id: content
        anchors.fill: parent
        parent: floating ? wndContent: root
        border.color: floatingWindowBorderColor
        color: floatingWindowBackgroundColor

        property var handle

        onHandleChanged: {
            checkHandle()
        }

        function checkHandle() {
            if(handle != undefined) {
                handle.parent  = content;
                handle.anchors.left = content.left
                handle.anchors.right = content.right
                handle.anchors.top = content.top
                handle.anchors.margins = 1
            }
            else {
//                console.log('Creating default handle for floating window ' + caption)
                handle = defaultHandleComponent.createObject(content)
                ma.z = handle.z+0.1
            }
        }

        MouseArea {
            id: ma
            anchors.fill: handle

            property var clickPos: "1,1"

            onPressed: {
                clickPos  = Qt.point(mouse.x,mouse.y)
                root.parent.somethingIsDragged = true;
                handleZ();
                //                console.log(root.parent.getWindowAt(mouse.x, mouse.y))

            }
            onReleased: {
                root.parent.somethingIsDragged = false;
            }

            onPositionChanged: {
                var delta = Qt.point(mouse.x-clickPos.x, mouse.y-clickPos.y)
                var p = root.floating ? window: root
                p.x += delta.x;
                p.y += delta.y;
                if(!floating) {

                    if(p.x < 0) p.x =  0;
                    if(p.y < 0) p.y =  0;
                    if(p.x > p.parent.width - p.width) p.x =  root.parent.width - p.width;
                    if(p.y > p.parent.height - p.height) p.y =  root.parent.height - p.height;
                }
            }
        }

        Item {
            clip: true
            id: area
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: handle != undefined ? handle.bottom: parent.top
            anchors.bottom: parent.bottom
            anchors.margins: 1
        }
    }
    Window {
        id: window
        width: root.width
        height: root.height
        visible: floating && !hidden
        flags: Qt.FramelessWindowHint
        Item {
            id: wndContent
            anchors.fill: parent
        }
        onVisibilityChanged: {
            //            switch(visibility) {
            //            case Window.Windowed: console.log('Windowed'); break;
            //            case Window.Maximized: console.log('Maximized'); break;
            //            case Window.Minimized: console.log('Minimized'); break;
            //            }
            if(visibility == Window.Windowed) {
                tmr.start()
            }
        }
        Timer {
            //to handle delays of window managers animations
            id: tmr
            interval: 500
            onTriggered: {
                switch(window.visibility) {
                case Window.Windowed: windowState = "windowed"; break;

                }

            }
        }
        onWidthChanged: {
            if(width < root.minimumWidth ) width = root.minimumWidth
            if(width > root.maximumWidth ) width = root.maximumWidth
        }

        onHeightChanged:{
            if(height < root.minimumHeight) height = root.minimumHeight
            if(height > root.maximumHeight) height = root.maximumHeight
            if(content.handle != undefined &&
                    height < content.handle.height + 2*content.border.width) height = content.handle.height + 2*content.border.width
        }
    }
    Component {
        id: defaultHandleComponent
        Rectangle {
            height: toolwindowHandleHeight

            gradient: Gradient {
                GradientStop { position: 0.0; color: "#505050" }
                GradientStop { position: 0.3; color: "#404040" }
                GradientStop { position: 1.0; color: "#434343" }
            }
			/*
            MCBasicText {
                color: _S.uiMainFontColor
                anchors.left: parent.left
                anchors.verticalCenter: parent.verticalCenter
                text: caption
                anchors.leftMargin: _S.uiSpacing
            }
			*/
            //            clip: true

        }
    }
    property bool readyForAnchor: !animationsActive && !floating
                                  && resizeMa.resizeProcState == -1
                                 && !ma.pressed && !parent.somethingIsDragged
    onReadyForAnchorChanged: {
        if (state == "maximized") return;
        var dist = mdiStickyDistance;
        if(readyForAnchor && x < dist)
            anchors.left = parent.left;
        if(!readyForAnchor)
            anchors.left = undefined;
        if(readyForAnchor && y < dist)
            anchors.top = parent.top;
        if(!readyForAnchor)
            anchors.top = undefined;
        if(readyForAnchor && x > parent.width - width - dist)
            anchors.right = parent.right;
        if(!readyForAnchor)
            anchors.right = undefined;
        if(readyForAnchor && y > parent.height - height - dist)
            anchors.bottom = parent.bottom;
        if(!readyForAnchor)
            anchors.bottom = undefined;
    }
    property bool animationsActive: liveView.visible
    Behavior on width {
        enabled: resizeMa.resizeProcState == -1 && !ma.pressed
        NumberAnimation {id: an1; duration: animationDelays; easing.type: Easing.InOutCubic}
    }
    Behavior on height {
        enabled: resizeMa.resizeProcState == -1 && !ma.pressed
        NumberAnimation {id: an2; duration: animationDelays; easing.type: Easing.InOutCubic}
    }
    Behavior on x {
        enabled: resizeMa.resizeProcState == -1 && !ma.pressed
        NumberAnimation {id: an3; duration: animationDelays; easing.type: Easing.InOutCubic}
    }
    Behavior on y {
        enabled: resizeMa.resizeProcState == -1 && !ma.pressed
        NumberAnimation {id: an4; duration: animationDelays; easing.type: Easing.InOutCubic}
    }


    property bool maximizable: true
    property bool minimizable: false
    property bool detachable: false
    onDetachableChanged: if(floating) floating = detachable
    onMaximizableChanged:if(windowState == "maximized") windowState = "windowed"
    onMinimizableChanged: if(windowState == "minimized") windowState = "windowed"
    function setDetached(detached) {
        if(detachable)
            floating = detached
    }
    function toggleDetached() {
        if(floating)
            setDetached(false)
        else
            setDetached(true)
    }


    function setMaximized(v) {
        if(maximizable) {
            if(v)
                windowState =  "maximized"
            else
                windowState = "windowed"
        }
    }
    function toggleMaximized() {
        if( windowState == "maximized") setMaximized(false)
        else setMaximized(true)
    }

    function setMinimized(v) {
        if(minimizable) {
            if(v)windowState =  "minimized"
            else windowState = "windowed"
        }
    }

    function toggleMinimized() {
        if( windowState == "minimized") setMinimized(false)
        else setMinimized(true)
    }
}

