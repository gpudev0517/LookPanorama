copy External\Libs\ffmpeg\windows\bin\x64\*.dll build\release\
copy External\Libs\libsourcey\vendor\bin\zlib.dll build\release\
copy External\Libs\opencv\x64\vc14\bin\release\*.dll build\release\
copy External\Binary\binaryForWebRTC\Release\*.dll build\release\
@rem copy External\Libs\webrtc-16937-win-x64\lib\x64\Release\*.dll build\release\

cd build\release\
C:\Qt\Qt5.7.0_64\5.7\msvc2015_64\bin\windeployqt.exe Look3D.exe
xcopy C:\Qt\Qt5.7.0_64\5.7\msvc2015_64\qml . /E /H /K