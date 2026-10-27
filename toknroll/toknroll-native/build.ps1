# libtoknroll on Windows: the Makefile's targets, in PowerShell (the shared half is ..\..\native.ps1).
#   .\build.ps1 library            -O2 -march=compatibility, what ships -> target\toknroll.dll (+ toknroll.lib to link against)
#   .\build.ps1 library -Quick     this machine, quick build (-Ob -march=native)
#   .\build.ps1 check              header vs exports, C example, Python tests
#   .\build.ps1 python             stage the DLL into python\toknroll\lib\windows-x86-64\
#   .\build.ps1 wheel              the wheel, tagged win_amd64 -> target\wheel\
#   .\build.ps1 check-wheel        twine check, a fresh venv, pip install, the tests from outside the tree
param([Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]] $Targets = @('help'), [switch] $Quick, [string] $Source = 'Qwen/Qwen3-0.6B')
$Here = $PSScriptRoot; $Name = 'toknroll'; $Opt = '-O2'; $March = 'compatibility'
$Headers = 'include\toknroll.h'; $Syms = 'toknroll'
$Example = {
    cl -nologo -std:c11 -W4 -Iinclude examples\c\tokenize.c -Fe:target\tokenize.exe -Fo:target\ target\toknroll.lib; if ($LASTEXITCODE) { throw "cl failed" }
    & target\tokenize.exe $Source "Hello, Tok'n'Roll!"; if ($LASTEXITCODE) { throw "tokenize failed" }
}
$PyTest = { Push-Location python; try { python -m unittest discover -s tests -t tests; if ($LASTEXITCODE) { throw "tests failed" } } finally { Pop-Location } }
$WheelDeps = { }
$WheelTest = { & wheelenv\Scripts\python -m unittest discover -s ..\python\tests -t ..\python\tests; if ($LASTEXITCODE) { throw "tests failed" } }
. "$PSScriptRoot\..\..\native.ps1"
