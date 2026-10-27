# libjinfer on Windows: the Makefile's targets, in PowerShell, since make is not native here.
#   .\build.ps1 library            -O3 -march=x86-64-v3, what ships -> target\jinfer.dll (+ jinfer.lib to link against)
#   .\build.ps1 library -Quick     this machine, quick build (-Ob -march=native)
#   .\build.ps1 check              headers vs exports, C example, Python checks
#   .\build.ps1 python             stage the DLL into python\jinfer\lib\windows-x86-64\
#   .\build.ps1 wheel              the wheel, tagged win_amd64 -> target\wheel\
#   .\build.ps1 check-wheel        twine check, a fresh venv with the toknroll wheel, pip install, the checks from outside the tree
#   .\build.ps1 publish            twine upload to PyPI after check-wheel; $env:TWINE_REPOSITORY = 'testpypi' rehearses
# Targets chain as in the Makefile: check-wheel builds the wheel, which stages the DLL.
param([Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]] $Targets = @('help'), [switch] $Quick)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Root = (Resolve-Path '..\..').Path
$Lib = 'target\jinfer.dll'
$ToknrollNative = "$Root\toknroll\toknroll-native"
$Platform = 'windows-x86-64'
$Tools = "$Root\target\pytools\" + (python -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep=chr(45))')   # chr(45) = "-": quotes do not survive PowerShell 5.1's argument passing

function Enter-Toolchain {   # cl, link, dumpbin for the shims, the image and the checks: Visual Studio's own environment
    if (Get-Command cl -ErrorAction SilentlyContinue) { return }
    $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
    Import-Module "$vs\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null
}
function Check-NativeImage {   # 25.0.4 is the floor: the isolate parameters the C shims use
    $v = (native-image --version | Select-Object -First 1) -replace '^native-image (\S+).*', '$1'
    if ([version](($v -split '\.')[0..2] -join '.') -lt [version]'25.0.4') { throw "native-image $v is too old: 25.0.4 or later" }   # 25.0.4.1.1: three parts compare
}
function Header-Functions { (Select-String -Path include\jinfer.h, "$ToknrollNative\include\toknroll.h" -Pattern '^(JINFER|TOKNROLL)_API.*?((jinfer|toknroll)_[a-z_]+)\(').Matches | ForEach-Object { $_.Groups[2].Value } | Sort-Object }
function Exports($dll) { dumpbin /nologo /EXPORTS $dll | ForEach-Object { if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)') { $Matches[1] } } }

function Build-Library {
    Enter-Toolchain; Check-NativeImage
    $opt = if ($Quick) { '-Ob' } else { '-O3' }; $march = if ($Quick) { 'native' } else { 'x86-64-v3' }
    Push-Location $Root
    # jam.native.skip: jam does not build with the MSVC toolchain yet; the image runs the Java kernels
    try { mvn -Pnative -pl jinfer/jinfer-native -am package -DskipTests "-Djam.native.skip=true" "-Djinfer.image.opt=$opt" "-Djinfer.image.march=$march"; if ($LASTEXITCODE) { throw "mvn failed" } }
    finally { Pop-Location }
}
function Check-Version {   # the header, the wheel and the pom name the same release (__version__ is the library's own string)
    $h = (Header-Version); $w = (Select-String python\pyproject.toml -Pattern '^version = "([^"]+)"').Matches[0].Groups[1].Value
    $m = (Select-String pom.xml -Pattern '<version>([^<]+)</version>' | Select-Object -First 1).Matches[0].Groups[1].Value
    if ($h -ne $w -or $h -ne $m) { throw "version drift: header $h, wheel $w, pom $m" }
    "version $h everywhere"
}
function Header-Version { ((Select-String include\jinfer.h -Pattern 'JINFER_VERSION_(MAJOR|MINOR|PATCH) (\d+)').Matches | ForEach-Object { $_.Groups[2].Value }) -join '.' }
function Check-Exports {   # every function the two headers declare is exported; the runtime's own names (graal_*, JNU_*) ride along on Windows
    Enter-Toolchain
    $header = Header-Functions; $exported = Exports $Lib | Where-Object { $_ -match '^(jinfer|toknroll)_[^_]' } | Sort-Object
    $diff = Compare-Object $header $exported
    if ($diff) { $diff | Format-Table | Out-String | Write-Host; throw "exports differ from the header" }
    "exports match the header: $($header.Count) functions"
}
function Run-Check {
    Check-Version; Check-Exports
    cl -nologo -W3 -Iinclude "-I$ToknrollNative\include" examples\c\chat.c -Fe:target\chat.exe -Fo:target\ target\jinfer.lib; if ($LASTEXITCODE) { throw "cl failed" }
    & target\chat.exe; if ($LASTEXITCODE) { throw "chat failed" }
    python python\test_jinfer.py; if ($LASTEXITCODE) { throw "checks failed" }
}
function Stage-Python { New-Item -ItemType Directory -Force "python\jinfer\lib\$Platform" | Out-Null; Copy-Item $Lib "python\jinfer\lib\$Platform\" }
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
function Check-Wheel {   # the wheel as a user gets it: metadata as PyPI renders it, installed into a fresh venv with the toknroll wheel it depends on, tested from outside the checkout
    Build-Wheel
    & "$ToknrollNative\build.ps1" wheel; Set-Location $PSScriptRoot
    $whl = (Get-Item target\wheel\*.whl).FullName; $toknroll = (Get-Item "$ToknrollNative\target\wheel\*.whl").FullName
    & "$Tools\Scripts\twine" check --strict $whl; if ($LASTEXITCODE) { throw "twine check failed" }
    Remove-Item -Recurse -Force target\wheelenv -ErrorAction SilentlyContinue; python -m venv target\wheelenv; & target\wheelenv\Scripts\pip install -q $whl $toknroll
    Push-Location target
    try { $env:JINFER_INSTALLED = '1'; & wheelenv\Scripts\python ..\python\test_jinfer.py; if ($LASTEXITCODE) { throw "checks failed" } }
    finally { Remove-Item Env:JINFER_INSTALLED; Pop-Location }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (-not ([IO.Compression.ZipFile]::OpenRead($whl).Entries.FullName -match 'LICENSE$')) { throw "no LICENSE in the wheel" }
    "the wheel carries the license"
}
function Publish { Check-Wheel; & "$Tools\Scripts\twine" upload (Get-Item target\wheel\*.whl).FullName; if ($LASTEXITCODE) { throw "upload failed" } }

foreach ($t in $Targets) {
    switch ($t) {
        'help'        { Get-Content $PSCommandPath | Select-Object -Skip 1 -First 8 | ForEach-Object { $_ -replace '^#\s?', '' } }
        'library'     { Build-Library }
        'check'       { Run-Check }
        'python'      { Stage-Python }
        'wheel'       { Build-Wheel }
        'check-wheel' { Check-Wheel }
        'publish'     { Publish }
        default       { throw "no target '$t'" }
    }
}
