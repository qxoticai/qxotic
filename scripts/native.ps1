# What the two native libraries' build.ps1 share: native.mk's targets in PowerShell, since make is not native
# on Windows. Dot-sourced by a build.ps1 that has set, first:
#   $Here        its own directory (a dot-sourced script's $PSScriptRoot is this file's)
#   $Name        the library: target\$Name.dll, include\$Name.h, the Python package python\$Name
#   $Headers     the headers whose functions the DLL exports, and $Syms, their prefixes
#   $Example     a script block: the C example, built and run
#   $PyTest      a script block: the Python tests, in the tree
#   $WheelTest   a script block: the tests from outside the tree, run in target\ with wheelenv's python
#   $WheelDeps   a script block: other wheels the installed one needs, built first
#   $WheelExample  the C example built against the installed wheel, and $WheelExampleArgs, how it is run
#   $Description the pkg-config file's Description line
#   $ExportHeaders, $ExportSyms, $ExtraExports  optional: the headers whose functions the DLL exports (default $Headers), their
#                prefixes (default $Syms), and exported names no header declares (libjam's JNI entry points)
#   $PgoProfile  optional, a script block: Oracle GraalVM's profile, recorded into target\pgo\default.iprof
#   $MvnArg      optional, one more argument for the library's Maven build, as its Makefile passes
$ErrorActionPreference = 'Stop'
Push-Location $Here   # the caller's location back at the end, as make -C leaves it
$Root = (Resolve-Path '..\..').Path
$Lib = "target\$Name.dll"
$Platform = 'windows-x86-64'
# chr(45) is "-": PowerShell 5.1 strips the quotes inside a native command's argument
$Tools = "$Root\target\pytools\" + (python -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep=chr(45))')

. "$Root\scripts\toolchain.ps1"   # Enter-Toolchain: cl, link, dumpbin for the shims, the image and the checks, clang for jam
function Check-NativeImage {   # 25.0.4 is the floor: the isolate parameters the C shims use
    $v = (native-image --version | Select-Object -First 1) -replace '^native-image (\S+).*', '$1'
    if ([version](($v -split '\.')[0..2] -join '.') -lt [version]'25.0.4') { throw "native-image $v is too old: 25.0.4 or later" }   # 25.0.4.1.1: [version] takes four parts at most
}
function Write-Exports {   # target\$Name.def, the API's module definition: the headers' functions and the extra names, for another toolchain's import library
    # (not a link input: the image's objects export the runtime's own names by directives of their own, and link.exe exports the union)
    $headers = if ($ExportHeaders) { $ExportHeaders } else { $Headers }; $syms = if ($ExportSyms) { $ExportSyms } else { $Syms }
    $names = (Select-String -Path $headers -Pattern "^[A-Z]+_API.*?(($syms)_[a-z0-9_]+)\(").Matches | ForEach-Object { $_.Groups[1].Value }
    $names = @($names) + @($ExtraExports) | Sort-Object -Unique
    New-Item -ItemType Directory -Force target | Out-Null
    [IO.File]::WriteAllText("$Here\target\$Name.def", (@("LIBRARY $Name", "EXPORTS") + $names -join "`r`n") + "`r`n")
    "$($names.Count) exports in target\$Name.def"
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
# The release as the header spells it, 0.4.0, or 0.4.0-rc1 for a release candidate, as native.mk's VERSION;
# Python's spelling drops the dash (0.4.0rc1, PEP 440's).
function Get-Version { (Select-String "include\$Name.h" -Pattern "^#define $($Name.ToUpper())_VERSION `"([^`"]+)`"").Matches[0].Groups[1].Value }
# The version the pom builds with, as native.mk's POM_VERSION and release-plan.sh read it: a pom's own <version>,
# outside <parent> and before any content, else the nearest enclosing directory's pom's.
function Get-PomVersion {
    for ($dir = $Here; Test-Path "$dir\pom.xml"; $dir = Split-Path $dir) {
        $pom = (Get-Content -Raw "$dir\pom.xml") -replace '(?s)<parent>.*?</parent>', ''
        $head = ($pom -split '<(?:properties|dependencies|dependencyManagement|build|modules|profiles)>')[0]
        if ($head -match '(?m)^  <version>([^<]+)</version>') { return $Matches[1] }
    }
}
function Check-Version {   # the header, its numbers, the pom, the wheel and (where it is a literal) the package name the same release
    $h = Get-Version; $y = $h -replace '-', ''
    $n = ((Select-String "include\$Name.h" -Pattern "$($Name.ToUpper())_VERSION_(MAJOR|MINOR|PATCH) (\d+)").Matches | ForEach-Object { $_.Groups[2].Value }) -join '.'
    $w = (Select-String python\pyproject.toml -Pattern '^version = "([^"]+)"').Matches[0].Groups[1].Value
    $m = Get-PomVersion
    $p = (Select-String "python\$Name\__init__.py" -Pattern '^__version__ = "([^"]+)"' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
    if ($h -notmatch "^$([regex]::Escape($n))(-rc\d+)?$") { throw "version drift: header $h, its numbers $n" }
    if ($h -ne $m -or $y -ne $w -or ($p -and $y -ne $p)) { throw "version drift: header $h, pom $m, wheel $w, package $p" }
    "version $h everywhere, $y for Python"
}
function Check-Exports {   # every function the headers declare is exported, and the module definition's names are all there is
    Enter-Toolchain
    $header = (Select-String -Path $Headers -Pattern "^[A-Z]+_API.*?(($Syms)_[a-z0-9_]+)\(").Matches | ForEach-Object { $_.Groups[1].Value } | Sort-Object
    $all = dumpbin /nologo /EXPORTS $Lib | ForEach-Object { if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)') { $Matches[1] } } | Sort-Object
    $exported = $all | Where-Object { $_ -match "^($Syms)_[^_]" }
    $diff = Compare-Object $header $exported
    if ($diff) { $diff | Format-Table | Out-String | Write-Host; throw "exports differ from the headers" }
    "$($all.Count - $exported.Count) exports beyond the headers, the runtime's own, as a Windows DLL has no export list"
    "exports match the headers: $($header.Count) functions"
}
function Run-Check { Check-Version; Check-Exports; & $Example; & $PyTest }
function Stage-Python {   # this platform's library, its import library and the headers, as native.mk's python
    Copy-Item "$Root\LICENSE", NOTICE python\   # pyproject's license-files: a pip install of python\ carries them, as the wheel does
    Remove-Item -Recurse -Force "python\$Name\lib", "python\$Name\include" -ErrorAction Ignore
    New-Item -ItemType Directory -Force "python\$Name\lib\$Platform", "python\$Name\include" | Out-Null
    Copy-Item $Lib, "target\$Name.lib" "python\$Name\lib\$Platform\"; Copy-Item $Headers "python\$Name\include\"
    # the pkg-config file, as native.mk writes it: paths relative to itself, so it is right wherever pip puts it
    $version = (Get-Version)
    New-Item -ItemType Directory -Force "python\$Name\lib\$Platform\pkgconfig" | Out-Null
    $pc = 'prefix=${pcfiledir}/../../..', 'includedir=${prefix}/include', 'libdir=${pcfiledir}/..', '',
          "Name: $Name", "Description: $Description", "Version: $version", 'Cflags: -I${includedir}', "Libs: -L`${libdir} -l$Name"
    [IO.File]::WriteAllText("$Here\python\$Name\lib\$Platform\pkgconfig\$Name.pc", ($pc -join "`n") + "`n")
}
function Ensure-Tools {   # one venv per Python, as native.mk's: pip builds the wheel, twine checks it
    if (Test-Path "$Tools\Scripts\twine.exe") { return }
    if (Test-Path $Tools) { Remove-Item -Recurse -Force $Tools }
    python -m venv $Tools; if ($LASTEXITCODE) { throw "venv failed" }
    & "$Tools\Scripts\pip" install -q twine; if ($LASTEXITCODE) { throw "pip install twine failed" }
}
function Build-Wheel {   # tagged py3-none-win_amd64 by setup.py: ctypes over a C library, any Python 3
    Stage-Python; Ensure-Tools
    Remove-Item -Recurse -Force target\wheel, python\build -ErrorAction Ignore
    & "$Tools\Scripts\pip" wheel -q --no-deps -w target\wheel .\python; if ($LASTEXITCODE) { throw "pip wheel failed" }
}
function Build-Dist {   # the release zip as native.mk's dist: a prefix tree, Windows' own layout (bin\ for the DLL, lib\ for the import library and the .def)
    Write-Exports | Out-Host
    $version = (Get-Version)
    $dist = "lib$Name-$version-$Platform"
    Remove-Item -Recurse -Force "target\dist\$dist", "target\dist\$dist.zip" -ErrorAction Ignore
    New-Item -ItemType Directory -Force "target\dist\$dist\include", "target\dist\$dist\lib", "target\dist\$dist\bin" | Out-Null
    Copy-Item $Headers "target\dist\$dist\include\"; Copy-Item $Lib "target\dist\$dist\bin\"; Copy-Item "target\$Name.lib", "target\$Name.def" "target\dist\$dist\lib\"   # the .def: another toolchain's import library
    Copy-Item README.md, NOTICE, "$Root\LICENSE" "target\dist\$dist\"
    Compress-Archive -Path "target\dist\$dist" -DestinationPath "target\dist\$dist.zip"
    $hash = (Get-FileHash "target\dist\$dist.zip" -Algorithm SHA256).Hash.ToLower()
    [IO.File]::WriteAllText("$Here\target\dist\$dist.zip.sha256", "$hash  $dist.zip`n")
    Get-Item "target\dist\$dist.zip" | Format-Table Name, Length | Out-String | Write-Host; Get-Content "target\dist\$dist.zip.sha256"
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
    foreach ($f in 'LICENSE', 'NOTICE') { if (-not (python -m zipfile -l $whl | Select-String $f)) { throw "no $f in the wheel" } }
    "the wheel carries the license and the notice"
    Enter-Toolchain   # cl, for the example: check-wheel alone has not built anything yet
    $cflags = & "target\wheelenv\Scripts\$Name-config.exe" --cflags; $libs = & "target\wheelenv\Scripts\$Name-config.exe" --libs   # one flag per line
    cl -nologo -std:c11 -W4 $cflags $WheelExample -Fe:target\wheel-example.exe -Fo:target\ /link $libs; if ($LASTEXITCODE) { throw "cl failed against the wheel" }
    $path = $env:PATH; $env:PATH = (& "target\wheelenv\Scripts\$Name-config.exe" --pkgconfigdir | Split-Path) + ";$env:PATH"   # the DLL beside the import library
    try { & target\wheel-example.exe @WheelExampleArgs | Out-Null; if ($LASTEXITCODE) { throw "the example failed against the wheel" } } finally { $env:PATH = $path }
    "the C example builds and runs against the installed wheel: $Name-config --cflags --libs"
}

try {
    foreach ($t in $Targets) {
        switch ($t) {
            'help'        { Get-Content "$Here\build.ps1" | Select-Object -Skip 1 | Select-Object -First 7 | ForEach-Object { $_ -replace '^#\s?', '' }; "This machine's two wheels from a clean export: ..\..\scripts\wheels.ps1." }
            'library'     { Build-Library }
            'check'       { Run-Check }
            'python'      { Stage-Python }
            'wheel'       { Build-Wheel }
            'check-wheel' { Check-Wheel }
            'dist'        { Build-Dist }
            default       { throw "no target '$t'" }
        }
    }
} finally { Pop-Location }
