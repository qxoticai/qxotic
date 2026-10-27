# What the two native libraries' build.ps1 share, the Makefile's targets in PowerShell since make is not
# native on Windows (the Makefiles' native.mk, here). Dot-sourced after the including script set $Here (its
# own directory), $Name, $Opt, $March, $Headers, $Syms and the script blocks $Example (the C example built and
# run), $PyTest (the tests in the tree), $WheelTest (the tests from outside it, run in target\ with wheelenv's
# python) and $WheelDeps (other wheels the installed one needs, built first).
$ErrorActionPreference = 'Stop'
Set-Location $Here
$Root = (Resolve-Path '..\..').Path
$Lib = "target\$Name.dll"
$Platform = 'windows-x86-64'
$Tools = "$Root\target\pytools\" + (python -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep=chr(45))')   # chr(45) = "-": quotes do not survive PowerShell 5.1's argument passing

function Enter-Toolchain {   # cl, link, dumpbin for the shims, the image and the checks: Visual Studio's own environment
    if (Get-Command cl -ErrorAction SilentlyContinue) { return }
    $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
    Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null
    # clang (jam builds on the MSVC ABI with it), cmake and ninja: Build Tools components, not on the dev shell's PATH
    $env:Path = "$vs\VC\Tools\Llvm\x64\bin;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$env:Path"
}
function Check-NativeImage {   # 25.0.4 is the floor: the isolate parameters the C shims use
    $v = (native-image --version | Select-Object -First 1) -replace '^native-image (\S+).*', '$1'
    if ([version](($v -split '\.')[0..2] -join '.') -lt [version]'25.0.4') { throw "native-image $v is too old: 25.0.4 or later" }   # 25.0.4.1.1: three parts compare
}
function Build-Library {
    Enter-Toolchain; Check-NativeImage
    $opt = if ($Quick) { '-Ob' } else { $Opt }; $march = if ($Quick) { 'native' } else { $March }
    Push-Location $Root
    try { mvn -Pnative -pl "$Name/$Name-native" -am package -DskipTests "-D$Name.image.opt=$opt" "-D$Name.image.march=$march"; if ($LASTEXITCODE) { throw "mvn failed" } }
    finally { Pop-Location }
}
function Check-Version {   # the header, the wheel, the pom and (where it is a literal) the package name the same release
    $h = ((Select-String "include\$Name.h" -Pattern "$($Name.ToUpper())_VERSION_(MAJOR|MINOR|PATCH) (\d+)").Matches | ForEach-Object { $_.Groups[2].Value }) -join '.'
    $w = (Select-String python\pyproject.toml -Pattern '^version = "([^"]+)"').Matches[0].Groups[1].Value
    $m = (Select-String pom.xml -Pattern '<version>([^<]+)</version>' | Select-Object -First 1).Matches[0].Groups[1].Value
    $p = (Select-String "python\$Name\__init__.py" -Pattern '^__version__ = "([^"]+)"' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
    if ($h -ne $w -or $h -ne $m -or ($p -and $h -ne $p)) { throw "version drift: header $h, wheel $w, pom $m, package $p" }
    "version $h everywhere"
}
function Check-Exports {   # every function the headers declare is exported; the runtime's own names (graal_*, JNU_*) ride along on Windows
    Enter-Toolchain
    $header = (Select-String -Path $Headers -Pattern "^[A-Z]+_API.*?(($Syms)_[a-z_]+)\(").Matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object
    $exported = dumpbin /nologo /EXPORTS $Lib | ForEach-Object { if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)') { $Matches[1] } } | Where-Object { $_ -match "^($Syms)_[^_]" } | Sort-Object
    $diff = Compare-Object $header $exported
    if ($diff) { $diff | Format-Table | Out-String | Write-Host; throw "exports differ from the headers" }
    "exports match the headers: $($header.Count) functions"
}
function Run-Check { Check-Version; Check-Exports; & $Example; & $PyTest }
function Stage-Python { New-Item -ItemType Directory -Force "python\$Name\lib\$Platform" | Out-Null; Copy-Item $Lib "python\$Name\lib\$Platform\" }
function Ensure-Tools {   # one venv per Python, as the Makefile: build, wheel, twine
    if (-not (Test-Path "$Tools\Scripts\twine.exe")) { Remove-Item -Recurse -Force $Tools -ErrorAction SilentlyContinue; python -m venv $Tools; & "$Tools\Scripts\pip" install -q build wheel twine }
}
function Build-Wheel {
    Stage-Python; Ensure-Tools
    Copy-Item "$Root\LICENSE" python\LICENSE
    Remove-Item -Recurse -Force target\wheel, python\build -ErrorAction SilentlyContinue
    & "$Tools\Scripts\python" -m build --wheel --outdir target\wheel python; if ($LASTEXITCODE) { throw "build failed" }
    & "$Tools\Scripts\wheel" tags --remove --python-tag py3 --abi-tag none --platform-tag win_amd64 (Get-Item target\wheel\*.whl).FullName; if ($LASTEXITCODE) { throw "wheel tags failed" }   # ctypes over a C library: any Python 3
}
function Check-Wheel {   # the wheel as a user gets it: metadata as PyPI renders it, installed into a fresh venv with the wheels it depends on, tested from outside the checkout
    Build-Wheel
    $whl = (Get-Item target\wheel\*.whl).FullName; $deps = & $WheelDeps; Set-Location $Here
    & "$Tools\Scripts\twine" check --strict $whl; if ($LASTEXITCODE) { throw "twine check failed" }
    Remove-Item -Recurse -Force target\wheelenv -ErrorAction SilentlyContinue; python -m venv target\wheelenv; & target\wheelenv\Scripts\pip install -q $whl $deps
    Push-Location target
    try { Set-Item "Env:$($Name.ToUpper())_INSTALLED" '1'; & $WheelTest } finally { Remove-Item "Env:$($Name.ToUpper())_INSTALLED"; Pop-Location }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (-not ([IO.Compression.ZipFile]::OpenRead($whl).Entries.FullName -match 'LICENSE$')) { throw "no LICENSE in the wheel" }
    "the wheel carries the license"
}

foreach ($t in $Targets) {
    switch ($t) {
        'help'        { Get-Content "$Here\build.ps1" | Select-Object -Skip 1 | Select-Object -First 7 | ForEach-Object { $_ -replace '^#\s?', '' }; 'All platforms at once, and PyPI: make wheels, make publish-wheels at the root.' }
        'library'     { Build-Library }
        'check'       { Run-Check }
        'python'      { Stage-Python }
        'wheel'       { Build-Wheel }
        'check-wheel' { Check-Wheel }
        default       { throw "no target '$t'" }
    }
}
