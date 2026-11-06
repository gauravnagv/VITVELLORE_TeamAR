@echo off
REM Opens Blender and plays the live eye-movement demo (about 75 seconds).
cd /d "%~dp0"
"%~dp0..\tools\blender-4.5.15-windows-x64\blender.exe" "%~dp0scene.blend" --python "%~dp0wrapper\demo_live.py"
