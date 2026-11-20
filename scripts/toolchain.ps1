# Visual Studio's own environment for cl, link and dumpbin, and the Build Tools' clang, cmake and ninja that are
# not on the dev shell's PATH: dot-sourced by the native build scripts, native.ps1 and jinfer\jinfer-cli\build.ps1.
function Enter-Toolchain {
    if (Get-Command cl -ErrorAction SilentlyContinue) { return }
    $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
    Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null
    $env:Path = "$vs\VC\Tools\Llvm\x64\bin;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$env:Path"
}

# The VC++ runtime a binary imports (vcruntime140*.dll), which Windows lacks: copied beside it in the release zips,
# app-local as Microsoft's Distributable Code allows (aka.ms/vs/17/redist.txt). Static /MT cannot replace it:
# GraalVM's own JDK libraries are /MD and import the CRT. dumpbin decides which, so a new import comes along.
function Copy-VcRuntime([string] $Binary, [string] $To) {
    Enter-Toolchain
    $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
    $crt = Get-ChildItem "$vs\VC\Redist\MSVC\1*\x64\Microsoft.VC*.CRT" -Directory | Sort-Object { [version]$_.Parent.Parent.Name } | Select-Object -Last 1
    if (-not $crt) { throw "no VC++ runtime under $vs\VC\Redist\MSVC: install the Build Tools' C++ workload" }
    $dlls = dumpbin /dependents $Binary | Select-String '^\s+(\S+\.dll)\s*$' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Where-Object { Test-Path "$($crt.FullName)\$_" }
    Copy-Item ($dlls | ForEach-Object { "$($crt.FullName)\$_" }) $To
}
# A zip as the format wants it: / between names (PS 5.1's Compress-Archive and .NET Framework's ZipFile write \),
# and no Unix modes, so an unzip elsewhere applies its own defaults (Windows' tar.exe records NTFS's 666 and 777)
function New-Zip([string] $Dir, [string] $Name) {
    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::Open("$Dir\$Name.zip", 'Create')
    try {
        foreach ($f in Get-ChildItem -Recurse -File "$Dir\$Name") {
            $entry = "$Name/" + $f.FullName.Substring("$Dir\$Name\".Length).Replace('\', '/')
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $entry, 'Optimal')
        }
    } finally { $zip.Dispose() }
}
