# libtoknroll on Windows: the Makefile's targets, in PowerShell (the shared half is ..\..\native.ps1).
#   .\build.ps1 library            what ships (Oracle GraalVM: PGO at -O3, a profile recorded first; CE: -O2), native-image's default target -> target\toknroll.dll (+ toknroll.lib)
#   .\build.ps1 library -Quick     this machine, quick build (-Ob -march=native)
#   .\build.ps1 check              header vs exports, C example, Python tests
#   .\build.ps1 python             stage the DLL into python\toknroll\lib\windows-x86-64\
#   .\build.ps1 wheel              the wheel, tagged win_amd64 -> target\wheel\
#   .\build.ps1 check-wheel        twine check, a fresh venv, pip install, the tests from outside the tree
param([Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]] $Targets = @('help'), [switch] $Quick, [string] $Source = 'Qwen/Qwen3-0.6B')
$Here = $PSScriptRoot; $Name = 'toknroll'
$Headers = 'include\toknroll.h'; $Syms = 'toknroll'
# Oracle GraalVM's profile, as the Makefile's pgo-profile: PgoTraining built instrumented with the library's own
# feature and options, and run on enwik8 -> target\pgo\default.iprof
$PgoProfile = {
    $pgo = "$Here\target\pgo"
    New-Item -ItemType Directory -Force $pgo | Out-Null; Remove-Item -Force -ErrorAction Ignore "$pgo\default.iprof"
    $script = "$Root\toknroll\scripts\download_enwik8.py"
    $corpus = python -c "import runpy; print(runpy.run_path(r'$script', run_name='lib')['default_corpus_dir']() / 'enwik8')"
    if (-not (Test-Path $corpus)) { python $script; if ($LASTEXITCODE) { throw "enwik8 failed" } }
    Push-Location $Root
    try { mvn -q -pl toknroll/toknroll-native -am test-compile dependency:build-classpath -DincludeScope=runtime "-Dmdep.outputFile=$pgo\classpath"; if ($LASTEXITCODE) { throw "mvn failed" } }
    finally { Pop-Location }
    cl -nologo -std:c11 -O2 -Iinclude -c src\main\c\toknroll_isolate.c "-Fo:$pgo\toknroll_isolate.obj"; if ($LASTEXITCODE) { throw "cl failed" }   # Workers' threads
    Push-Location $pgo
    try {
        # quoted: PowerShell 5.1 splits an unquoted -argument at its first dot
        native-image --pgo-instrument "-H:NativeLinkerOption=$pgo\toknroll_isolate.obj" `
            -cp "$Here\target\classes;$Here\target\test-classes;$(Get-Content classpath)" com.qxotic.toknroll.libtoknroll.PgoTraining -o train
        if ($LASTEXITCODE) { throw "native-image failed" }
        & .\train.exe $corpus; if ($LASTEXITCODE) { throw "training failed" }
    } finally { Pop-Location }
}
$Example = {
    cl -nologo -std:c11 -W4 -Iinclude examples\c\tokenize.c -Fe:target\tokenize.exe -Fo:target\ target\toknroll.lib; if ($LASTEXITCODE) { throw "cl failed" }
    & target\tokenize.exe $Source "Hello, Tok'n'Roll!"; if ($LASTEXITCODE) { throw "tokenize failed" }
}
$PyTest = { Push-Location python; try { python -m unittest discover -s tests -t tests; if ($LASTEXITCODE) { throw "tests failed" } } finally { Pop-Location } }
$WheelDeps = { }
$WheelTest = { & wheelenv\Scripts\python -m unittest discover -s ..\python\tests -t ..\python\tests; if ($LASTEXITCODE) { throw "tests failed" } }
. "$PSScriptRoot\..\..\native.ps1"
