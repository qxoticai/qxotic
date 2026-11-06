# Visual Studio's own environment for cl, link and dumpbin, and the Build Tools' clang, cmake and ninja that are
# not on the dev shell's PATH: dot-sourced by the native build scripts, native.ps1 and jinfer\jinfer-cli\build.ps1.
function Enter-Toolchain {
    if (Get-Command cl -ErrorAction SilentlyContinue) { return }
    $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
    Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null
    $env:Path = "$vs\VC\Tools\Llvm\x64\bin;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$env:Path"
}
