# libtoknroll on Windows: the Makefile's targets, in PowerShell, since make is not native here.
#   .\build.ps1 library            -O2 -march=compatibility, what ships -> target\toknroll.dll (+ toknroll.lib to link against)
#   .\build.ps1 library -Quick     this machine, quick build (-Ob -march=native)
#   .\build.ps1 check              header vs exports, C example, Python tests
#   .\build.ps1 python             stage the DLL into python\toknroll\lib\windows-x86-64\
#   .\build.ps1 wheel              the wheel, tagged win_amd64 -> target\wheel\
#   .\build.ps1 check-wheel        twine check, a fresh venv, pip install, the suite from outside the tree
#   .\build.ps1 publish            twine upload to PyPI after check-wheel; $env:TWINE_REPOSITORY = 'testpypi' rehearses
# Targets chain as in the Makefile: check-wheel builds the wheel, which stages the DLL.
param([Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]] $Targets = @('help'), [switch] $Quick, [string] $Source = 'Qwen/Qwen3-0.6B')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Root = (Resolve-Path '..\..').Path
$Lib = 'target\toknroll.dll'
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
function Header-Functions { (Select-String -Path include\toknroll.h -Pattern '^TOKNROLL_API.*?(toknroll_[a-z_]+)\(').Matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object }
function Exports($dll) { dumpbin /nologo /EXPORTS $dll | ForEach-Object { if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)') { $Matches[1] } } }

function Build-Library {
    Enter-Toolchain; Check-NativeImage
    $opt = if ($Quick) { '-Ob' } else { '-O2' }; $march = if ($Quick) { 'native' } else { 'compatibility' }
    Push-Location $Root
    try { mvn -Pnative -pl toknroll/toknroll-native -am package -DskipTests "-Dtoknroll.image.opt=$opt" "-Dtoknroll.image.march=$march"; if ($LASTEXITCODE) { throw "mvn failed" } }
    finally { Pop-Location }
}
function Check-Version {   # the header, the wheel, the package and the pom name the same release
    $h = (Header-Version); $w = (Select-String python\pyproject.toml -Pattern '^version = "([^"]+)"').Matches[0].Groups[1].Value
    $p = (Select-String python\toknroll\__init__.py -Pattern '^__version__ = "([^"]+)"').Matches[0].Groups[1].Value
    $m = (Select-String pom.xml -Pattern '<version>([^<]+)</version>' | Select-Object -First 1).Matches[0].Groups[1].Value
    if ($h -ne $w -or $h -ne $p -or $h -ne $m) { throw "version drift: header $h, wheel $w, package $p, pom $m" }
    "version $h everywhere"
}
function Header-Version { ((Select-String include\toknroll.h -Pattern 'TOKNROLL_VERSION_(MAJOR|MINOR|PATCH) (\d+)').Matches | ForEach-Object { $_.Groups[2].Value }) -join '.' }
function Check-Exports {   # every function the header declares is exported; the runtime's own names (graal_*, JNU_*) ride along on Windows
    Enter-Toolchain
    $header = Header-Functions; $exported = Exports $Lib | Where-Object { $_ -match '^toknroll_[^_]' } | Sort-Object
    $diff = Compare-Object $header $exported
    if ($diff) { $diff | Format-Table | Out-String | Write-Host; throw "exports differ from the header" }
    "exports match the header: $($header.Count) functions"
}
function Run-Check {
    Check-Version; Check-Exports
    cl -nologo -W3 -Iinclude examples\c\tokenize.c -Fe:target\tokenize.exe -Fo:target\ target\toknroll.lib; if ($LASTEXITCODE) { throw "cl failed" }
    & target\tokenize.exe $Source "Hello, Tok'n'Roll!"; if ($LASTEXITCODE) { throw "tokenize failed" }
    Push-Location python; try { python -m unittest discover -s tests -t tests; if ($LASTEXITCODE) { throw "tests failed" } } finally { Pop-Location }
}
function Stage-Python { New-Item -ItemType Directory -Force "python\toknroll\lib\$Platform" | Out-Null; Copy-Item $Lib "python\toknroll\lib\$Platform\" }
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
function Check-Wheel {   # the wheel as a user gets it: metadata as PyPI renders it, installed into a fresh venv, tested from outside the checkout
    Build-Wheel
    $whl = (Get-Item target\wheel\*.whl).FullName
    & "$Tools\Scripts\twine" check --strict $whl; if ($LASTEXITCODE) { throw "twine check failed" }
    Remove-Item -Recurse -Force target\wheelenv -ErrorAction SilentlyContinue; python -m venv target\wheelenv; & target\wheelenv\Scripts\pip install -q $whl
    Push-Location target
    try { $env:TOKNROLL_INSTALLED = '1'; & wheelenv\Scripts\python -m unittest discover -s ..\python\tests -t ..\python\tests; if ($LASTEXITCODE) { throw "tests failed" } }
    finally { Remove-Item Env:TOKNROLL_INSTALLED; Pop-Location }
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
