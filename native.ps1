# What the two native libraries' build.ps1 share: native.mk's targets in PowerShell, since make is not native
# on Windows. Dot-sourced by a build.ps1 that has set, first:
#   $Here        its own directory (a dot-sourced script's $PSScriptRoot is this file's)
#   $Name        the library: target\$Name.dll, include\$Name.h, the Python package python\$Name
#   $Headers     the headers whose functions the DLL exports, and $Syms, their prefixes
#   $Example     a script block: the C example, built and run
#   $PyTest      a script block: the Python tests, in the tree
#   $WheelTest   a script block: the tests from outside the tree, run in target\ with wheelenv's python
#   $WheelDeps   a script block: other wheels the installed one needs, built first
#   $PgoProfile  optional, a script block: Oracle GraalVM's profile, recorded into target\pgo\default.iprof
#   $MvnArg      optional, one more argument for the library's Maven build, as its Makefile passes
$ErrorActionPreference = 'Stop'
Push-Location $Here   # the caller's location back at the end, as make -C leaves it
$Root = (Resolve-Path '..\..').Path
$Lib = "target\$Name.dll"
$Platform = 'windows-x86-64'
# chr(45) is "-": PowerShell 5.1 strips the quotes inside a native command's argument
$Tools = "$Root\target\pytools\" + (python -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep=chr(45))')

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
    if ([version](($v -split '\.')[0..2] -join '.') -lt [version]'25.0.4') { throw "native-image $v is too old: 25.0.4 or later" }   # 25.0.4.1.1: [version] takes four parts at most
}
function Build-Library {   # the pom's optimization unless -Quick (-Ob -march=native) or a PGO profile asks for another
    Enter-Toolchain; Check-NativeImage
    $options = $env:NATIVE_IMAGE_OPTIONS
    if ($Quick) { $env:NATIVE_IMAGE_OPTIONS = "$options -march=native".Trim() }
    try {
        # Oracle GraalVM builds from the profile $PgoProfile records first, for a library that has one
        $pgo = -not $Quick -and $PgoProfile -and ((native-image --version | Select-Object -Index 1) -match 'Oracle GraalVM')
        if ($pgo) { & $PgoProfile | Out-Host }
        # $null passes no argument, so the pom's stays; a splatted array would not do: 5.1 splits its -name:value elements
        $opt = if ($Quick) { "-D$Name.image.opt=-Ob" } elseif ($pgo) { "-D$Name.image.opt=--pgo=$Here\target\pgo\default.iprof" }
        Push-Location $Root
        try { mvn -Pnative -pl "$Name/$Name-native" -am package -DskipTests $opt $MvnArg; if ($LASTEXITCODE) { throw "mvn failed" } }
        finally { Pop-Location }
    } finally { $env:NATIVE_IMAGE_OPTIONS = $options }
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
    $header = (Select-String -Path $Headers -Pattern "^[A-Z]+_API.*?(($Syms)_[a-z0-9_]+)\(").Matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object
    $exported = dumpbin /nologo /EXPORTS $Lib | ForEach-Object { if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)') { $Matches[1] } } | Where-Object { $_ -match "^($Syms)_[^_]" } | Sort-Object
    $diff = Compare-Object $header $exported
    if ($diff) { $diff | Format-Table | Out-String | Write-Host; throw "exports differ from the headers" }
    "exports match the headers: $($header.Count) functions"
}
function Run-Check { Check-Version; Check-Exports; & $Example; & $PyTest }
function Stage-Python {   # this platform's library alone, as native.mk's python
    Remove-Item -Recurse -Force "python\$Name\lib" -ErrorAction Ignore
    New-Item -ItemType Directory -Force "python\$Name\lib\$Platform" | Out-Null; Copy-Item $Lib "python\$Name\lib\$Platform\"
}
function Ensure-Tools {   # one venv per Python, as native.mk's: pip builds the wheel, twine checks it
    if (Test-Path "$Tools\Scripts\twine.exe") { return }
    if (Test-Path $Tools) { Remove-Item -Recurse -Force $Tools }
    python -m venv $Tools; if ($LASTEXITCODE) { throw "venv failed" }
    & "$Tools\Scripts\pip" install -q twine; if ($LASTEXITCODE) { throw "pip install twine failed" }
}
function Build-Wheel {   # tagged py3-none-win_amd64 by setup.py: ctypes over a C library, any Python 3
    Stage-Python; Ensure-Tools
    Copy-Item "$Root\LICENSE" python\LICENSE
    Remove-Item -Recurse -Force target\wheel, python\build -ErrorAction Ignore
    & "$Tools\Scripts\pip" wheel -q --no-deps -w target\wheel .\python; if ($LASTEXITCODE) { throw "pip wheel failed" }
}
function Check-Wheel {   # the wheel as a user gets it: metadata as PyPI renders it, installed into a fresh venv with the wheels it depends on, tested from outside the checkout
    Build-Wheel
    $whl = (Get-Item target\wheel\*.whl).FullName; $deps = & $WheelDeps
    & "$Tools\Scripts\twine" check --strict $whl; if ($LASTEXITCODE) { throw "twine check failed" }
    if (Test-Path target\wheelenv) { Remove-Item -Recurse -Force target\wheelenv }
    python -m venv target\wheelenv; if ($LASTEXITCODE) { throw "venv failed" }
    & target\wheelenv\Scripts\pip install -q $whl $deps; if ($LASTEXITCODE) { throw "pip install failed" }
    Push-Location target
    try { Set-Item "Env:$($Name.ToUpper())_INSTALLED" '1'; & $WheelTest } finally { Remove-Item "Env:$($Name.ToUpper())_INSTALLED"; Pop-Location }
    if (-not (python -m zipfile -l $whl | Select-String 'LICENSE')) { throw "no LICENSE in the wheel" }
    "the wheel carries the license"
}

try {
    foreach ($t in $Targets) {
        switch ($t) {
            'help'        { Get-Content "$Here\build.ps1" | Select-Object -Skip 1 | Select-Object -First 6 | ForEach-Object { $_ -replace '^#\s?', '' }; "This machine's two wheels from a clean export: ..\..\wheels.ps1." }
            'library'     { Build-Library }
            'check'       { Run-Check }
            'python'      { Stage-Python }
            'wheel'       { Build-Wheel }
            'check-wheel' { Check-Wheel }
            default       { throw "no target '$t'" }
        }
    }
} finally { Pop-Location }
