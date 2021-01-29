
function getDefaultGeometries() {
    return {

    };
}
var defaultVideoWIndowsArea = {l: 425, t:0, r:0, b:425};
var defaultGeomData_d = ({});
var defaultGeomData_l = ({});
var defaultGeomData_p = ({});
function toolWindowGeometries() {
    if (Object.keys(defaultGeomData_d).length == 0) {
        defaultGeomData_d[cameraSettings.toolWindowTextID] =  {
            i: cameraSettings,
            width: 261,
            a: { 'top': 'border', 'left': 'border', 'bottom': 'border' },
        }
        defaultGeomData_d[takeManagement.toolWindowTextID] = {
            i: takeManagement,
            height: 250,
            width: 261,
            a: { 'bottom': 'border', 'left': cameraSettings.toolWindowTextID, 'bottom': transportControls.toolWindowTextID },
        }
        defaultGeomData_d[cameraManagement.toolWindowTextID] = {
            i: cameraManagement,
            height: 280,
            width: 640,
            a: { 'bottom': 'border', 'left': takeManagement.toolWindowTextID, 'bottom': transportControls.toolWindowTextID },
        }

        defaultGeomData_d[transportControls.toolWindowTextID] = {
            i: transportControls,
            height: 250,
            a: { 'bottom': 'border', 'right': 'border', 'left': cameraSettings.toolWindowTextID },
        }
        defaultGeomData_d[timecodeDisplay.toolWindowTextID] =    {
            i: timecodeDisplay,
            width: 300,
            height: 250,
            a: {'bottom':'border', 'right':'border'},
        }
    }
    return defaultGeomData_d;
}

function setToolWindowsGeom(wndList) {
    var toolWindowGeometriesData = toolWindowGeometries();
    for(var i = 0; i < wndList.length; i ++) {
        var d = toolWindowGeometriesData[wndList[i].toolWindowTextID];
        if(d != undefined) {
            if(d.width != undefined) {
                d.i.width = d.width;
            }
            if(d.height != undefined) {
                d.i.height = d.height;
            }
           
            var l = search(toolWindowGeometriesData, d.a['left'], 'left');
            var r = search(toolWindowGeometriesData, d.a['right'], 'right');
            var t = search(toolWindowGeometriesData, d.a['top'], 'top');
            var b = search(toolWindowGeometriesData, d.a['bottom'], 'bottom');

            var w_ = 0;
            var h_ = 0;
            if(d.width != undefined) {
                d.i.width = d.width;
                w_ = d.width;
            }
            else {
                w_ = d.i.parent.width - l - r;
                if(w_ < 10) w_ = 10;
                d.i.width = w_;
            }
            if(d.height != undefined) {
                d.i.height = d.height;
                h_ = d.height
            }
            else {
                h_ = d.i.parent.height - t - b;
                if(h_ < 10) h_ = 10;
                d.i.height = h_;
            }
                                    
            if (l != undefined) {
                wndList[i].x = l;
            } else if (r != undefined)
                wndList[i].x = wndList[i].parent.width - r - w_;
            if(t != undefined)
                wndList[i].y = t;
            else if(b != undefined)
                wndList[i].y = wndList[i].parent.height - b - h_;
        }
    }
}

function liveGeometries() {
    if (Object.keys(defaultGeomData_l).length == 0) {
        defaultGeomData_l[cameraSettings.toolWindowTextID] =  {
            i: cameraSettings,
            width: 280,
            a: { 'top': 'border', 'left': 'border', 'bottom': 'border' },
        }

        defaultGeomData_l[transportControls.toolWindowTextID] = {
            i: transportControls,
            height: 250,
            a: { 'bottom': 'border', 'right': timecodeDisplay.toolWindowTextID, 'left': cameraSettings.toolWindowTextID },
        }

        defaultGeomData_l[cameraManagement.toolWindowTextID] = {
            i: cameraManagement,
            height: 280,
            width: 640,
            a: { 'bottom': 'border', 'left': cameraSettings.toolWindowTextID },
        }

        defaultGeomData_l[nameChangeView.toolWindowTextID] = {
            i: nameChangeView,
            height: 100,
            width: 200,
            a: { 'bottom': 'border', 'left': cameraManagement.toolWindowTextID },
        }

        defaultGeomData_l[messageBox.toolWindowTextID] = {
            i: messageBox,
            width: 368,
            height: 185,
            a: { 'bottom': 'border' ,'left': cameraSettings.toolWindowTextID},
            a: { 'left': cameraManagement.toolWindowTextID, 'bottom': transportControls.toolWindowTextID },
        }
    }
    return defaultGeomData_l;
}

function setLiveGeom(wndList) {
    var toolWindowGeometriesData = liveGeometries();
    for (var i = 0; i < wndList.length; i++) {
        var d = toolWindowGeometriesData[wndList[i].toolWindowTextID];
        if (d != undefined) {
            if (d.width != undefined) {
                d.i.width = d.width;
            }
            if (d.height != undefined) {
                d.i.height = d.height;
            }

            var l = search(toolWindowGeometriesData, d.a['left'], 'left');
            var r = search(toolWindowGeometriesData, d.a['right'], 'right');
            var t = search(toolWindowGeometriesData, d.a['top'], 'top');
            var b = search(toolWindowGeometriesData, d.a['bottom'], 'bottom');

            var w_ = 0;
            var h_ = 0;
            if (d.width != undefined) {
                d.i.width = d.width;
                w_ = d.width;
            }
            else {
                w_ = d.i.parent.width - l - r;
                if (w_ < 10) w_ = 10;
                d.i.width = w_;
            }
            if (d.height != undefined) {
                d.i.height = d.height;
                h_ = d.height
            }
            else {
                h_ = d.i.parent.height - t - b;
                if (h_ < 10) h_ = 10;
                d.i.height = h_;
            }

            if (l != undefined) {
                wndList[i].x = l;
            } else if (r != undefined)
                wndList[i].x = wndList[i].parent.width - r - w_;
            if (t != undefined)
                wndList[i].y = t;
            else if (b != undefined)
                wndList[i].y = wndList[i].parent.height - b - h_;
        }
    }
    /*var end = wndList.length - 1;
    if(wndList[end] == messageBox)
    {
       wndList[end].x = wndList[end].parent.width /2 ;
       wndList[end].y = wndList[end].parent.height / 2;
    }*/
}

function playBackGeometries() {
    if (Object.keys(defaultGeomData_p).length == 0) {
        defaultGeomData_p[cameraSettings.toolWindowTextID] = {
            i: cameraSettings,
            width: 280,
            a: { 'top': 'border', 'left': 'border', 'bottom': 'border' },
        }

        defaultGeomData_p[transportControls.toolWindowTextID] = {
            i: transportControls,
            height: 250,
            a: { 'bottom': 'border', 'right': 'border', 'left': cameraSettings.toolWindowTextID },
        }
        
        defaultGeomData_p[cameraManagement.toolWindowTextID] = {
            i: cameraManagement,
            height: 280,
            width: 640,
            a: { 'bottom': 'border', 'left': cameraSettings.toolWindowTextID },
        }

        defaultGeomData_p[nameChangeView.toolWindowTextID] = {
            i: nameChangeView,
            height: 100,
            width: 200,
            a: { 'bottom': transportControls.toolWindowTextID, 'left': cameraManagement.toolWindowTextID },
        }
        defaultGeomData_p[messageBox.toolWindowTextID] = {
            i: messageBox,
            width: 368,
            height: 185,
             a: { 'bottom': 'border','left': cameraSettings.toolWindowTextID},
            a: { 'left': cameraSettings.toolWindowTextID, 'bottom': transportControls.toolWindowTextID },
        }
    }
    return defaultGeomData_p;
}

function setplayBackGeom(wndList) {
    var toolWindowGeometriesData = playBackGeometries();
    for (var i = 0; i < wndList.length; i++) {
        var d = toolWindowGeometriesData[wndList[i].toolWindowTextID];
        if (d != undefined) {
            if (d.width != undefined) {
                d.i.width = d.width;
            }
            if (d.height != undefined) {
                d.i.height = d.height;
            }

            var l = search(toolWindowGeometriesData, d.a['left'], 'left');
            var r = search(toolWindowGeometriesData, d.a['right'], 'right');
            var t = search(toolWindowGeometriesData, d.a['top'], 'top');
            var b = search(toolWindowGeometriesData, d.a['bottom'], 'bottom');

            var w_ = 0;
            var h_ = 0;
            if (d.width != undefined) {
                d.i.width = d.width;
                w_ = d.width;
            }
            else {
                w_ = d.i.parent.width - l - r;
                if (w_ < 10) w_ = 10;
                d.i.width = w_;
            }
            if (d.height != undefined) {
                d.i.height = d.height;
                h_ = d.height
            }
            else {
                h_ = d.i.parent.height - t - b;
                if (h_ < 10) h_ = 10;
                d.i.height = h_;
            }

            if (l != undefined) {
                wndList[i].x = l;
            } else if (r != undefined)
                wndList[i].x = wndList[i].parent.width - r - w_;
            if (t != undefined)
                wndList[i].y = t;
            else if (b != undefined)
                wndList[i].y = wndList[i].parent.height - b - h_;
        }
    }
    /*var end = wndList.length - 1;
    if(wndList[end] == messageBox)
    {
       wndList[end].x = wndList[end].parent.width /2 ;
       wndList[end].y = wndList[end].parent.height / 2;
    }*/
}

function    setGeom_anc( wndList) {
    var toolWindowGeometriesData = toolWindowGeometries()
    for(var i = 0; i < wndList.length; i ++) {
        var d = toolWindowGeometriesData[wndList[i].toolWindowTextID];
        if(d != undefined) {
            if(d.width != undefined) {
                d.i.width = d.width;
            }
            if(d.height != undefined) {
                d.i.height = d.height;
            }

            function getAnchor(d, anchorName) {
                if(d.a[anchorName] == 'border') {
                    return d.i.parent[anchorName];
                }
                else {
                    return toolWindowGeometriesData[d.a[anchorName]].i[({left:'right', right: 'left', top:'bottom', bottom:'top'})[anchorName]];
                }
            }

            for(var anchorName in d.a) {

                if(d.a[anchorName] == 'border') {
                    d.i.anchors[anchorName] =  d.i.parent[anchorName];
                }
                else {
                    d.i.anchors[anchorName] =  toolWindowGeometriesData[d.a[anchorName]].i[({left:'right', right: 'left', top:'bottom', bottom:'top'})[anchorName]];
                    d.i.anchors[anchorName+'Margin'] = -1;
                }
            }
        }
    }
}


function search(geomDescr, id, dir) {
    if(id == undefined) {
        return undefined;
    }
    if(id == 'border') return 0;
    var i = 0;
    var s;
    if(dir == 'left' || dir == 'right') s = 'width';
    else if (dir == 'top' || dir == 'bottom') s = 'height';
    else return 0;

    if(geomDescr[id] !=undefined) {

        if(geomDescr[id].a[dir] == undefined) {
            console.log(id, dir, undefined);
            return undefined;
        }
        return geomDescr[id][s] + search(geomDescr, geomDescr[id].a[dir], dir) -1
    }
}

function removeA(arr) {
    var what, a = arguments, L = a.length, ax;
    while (L > 1 && arr.length) {
        what = a[--L];
        while ((ax= arr.indexOf(what)) !== -1) {
            arr.splice(ax, 1);
        }
    }
    return arr;
}

function setVideoWindowsGeom(wndList, priorityWnd) {
    var windows = [];
    console.log("load position of the CameraView")
    for(var i = 0; i < wndList.length; i ++) {
        console.log("wndList.length of the CameraView 1 :", wndList.length)
        if (wndList[i].windowMenuGroup == 'video' && wndList[i].hidden == false) {
            windows.push(wndList[i]);
        }
    }
    console.log("wndList.length of the CameraView 2 :", windows.length)
    if(windows.length > 0) {
        var priorIndex = windows.indexOf(priorityWnd);
        if(priorIndex != -1) {
            var t = windows[0];
            windows[0] = windows[priorIndex];
            windows[priorIndex] = t;
        }

        var cellsCount = windows.length;
        var wScreen = windows[0].parent.width - defaultVideoWIndowsArea.l - defaultVideoWIndowsArea.r
        var hScreen = windows[0].parent.height - defaultVideoWIndowsArea.t - defaultVideoWIndowsArea.b

        var ratio = wScreen/hScreen;
        var vCount =Math.sqrt(cellsCount/ratio)
        var hCount =vCount*ratio;
        hCount = Math.round(hCount);
        if(hCount * Math.round(vCount) >= cellsCount)
            vCount = Math.round(vCount);
        else
            vCount = Math.round(vCount + 1);
        var winW = wScreen/hCount;
        var winH = hScreen/vCount;
        var doneList = [];
        for(var j = 0; j < vCount * hCount; j ++) {
            doneList.push(j)

        }



        for(var windowIndex = 0; windowIndex < windows.length; windowIndex++) {
            var closestIndex = doneList[0];

            var hIndex = closestIndex % hCount;
            var vIndex = (closestIndex - hIndex) / hCount;
            var closestDist = Math.sqrt(Math.pow(windows[windowIndex].x + windows[windowIndex].width/2 - winW * (hIndex + 0.5) - defaultVideoWIndowsArea.l, 2) +
                               Math.pow(windows[windowIndex].y + windows[windowIndex].height/2 - winH* (vIndex + 0.5) - defaultVideoWIndowsArea.t, 2));
            for(var j = 0; j < doneList.length; j ++) {
                hIndex = doneList[j] % hCount;
                vIndex = (doneList[j] - hIndex) / hCount;

                var dist = Math.sqrt(Math.pow(windows[windowIndex].x + windows[windowIndex].width/2 - winW * (hIndex + 0.5) - defaultVideoWIndowsArea.l, 2) +
                            Math.pow(windows[windowIndex].y + windows[windowIndex].height/2 - winH* (vIndex + 0.5) - defaultVideoWIndowsArea.t, 2));
                if(dist < closestDist){
                    closestIndex = doneList[j];
                    closestDist = dist;
                }
            }
            hIndex = closestIndex % hCount;
            vIndex = (closestIndex - hIndex) / hCount;

            windows[windowIndex].width = winW;
            windows[windowIndex].height = winH;
            windows[windowIndex].x = /*defaultVideoWIndowsArea.l + winW * 0.35 + 130 * windowIndex*/hIndex * winW + defaultVideoWIndowsArea.l;
            windows[windowIndex].y = /*defaultVideoWIndowsArea.t + winH * 0.6*/vIndex * winH + defaultVideoWIndowsArea.t;
            console.log("windows[",windowIndex,"].x =",windows[windowIndex].x);
            console.log("windows[",windowIndex,"].y =",windows[windowIndex].y);
            removeA(doneList, closestIndex)
        }
    }
}
