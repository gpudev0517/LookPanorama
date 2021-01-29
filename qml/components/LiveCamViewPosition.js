function setVideoWindowsGeom(wndList, parentWnd, width, height) {

    var windows = [];
    var selectedWindowPos = [] ;

    for(var i = 0; i < wndList.length; i ++) {
        windows.push(wndList[i]);
    }

    var wScreen = parentWnd.width - 20;
    var winW = width + 20;
    var winH = height + 20;
    var hCount = Math.floor(wScreen / winW);

    for(var windowIndex = 0; windowIndex < windows.length ; windowIndex++) {

        windows[windowIndex].width = width;
        windows[windowIndex].height = height;

        var hIndex = windowIndex % hCount;
        var vIndex = Math.floor(windowIndex / hCount);

        windows[windowIndex].x = hIndex * width + 20*(hIndex + 1)
        windows[windowIndex].y = vIndex * height + 20*(vIndex + 1)

        if (liveCam_index == windowIndex){
            windows[windowIndex].win_x = hIndex * width + 20*(hIndex + 1)
            windows[windowIndex].win_y = vIndex * height + 20*(vIndex + 1)
        }
    }
}

function getLiveAreaGeom(wndCount,parentWnd, width, height, margin) {

    var wScreen = parentWnd.width - margin;
    var winW = width + margin;
    var winH = height + margin;
    var hCount = Math.floor(wScreen / winW);

    var hIndex = 0;
    var vIndex = 0;

    for(var windowIndex = 0; windowIndex < wndCount ; windowIndex++) {

        hIndex = windowIndex % hCount;
        vIndex = Math.floor(windowIndex / hCount);

        var x = hIndex * width + 20*(hIndex + 1)
        var y = vIndex * height + 20*(vIndex + 1)
    }

    var wndAreaGeom = [];
    wndAreaGeom[0] = (hIndex + 1) * winW; wndAreaGeom[1] = (vIndex + 1) * winH;

    return wndAreaGeom;
}

 function setSelectedLiveCamViewGeom(wndList, parentWnd) {

    var windows = [];

    for (var i = 0; i < wndList.length; i ++) {
        windows.push(wndList[i]);
    }

    for (var windowIndex = 0; windowIndex < windows.length; windowIndex ++) {

        if (windows[windowIndex].z == 2) {
            windows[windowIndex].x = 0;
            windows[windowIndex].y = 0;
            windows[windowIndex].width = parentWnd.width;
            windows[windowIndex].height = parentWnd.height;
            break;
        }
    }
}


