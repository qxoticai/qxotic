# libjinfer on Windows: the Makefile's targets, in PowerShell (the shared half is ..\..\native.ps1).
#   .\build.ps1 library            -O3 -march=x86-64-v3, what ships -> target\jinfer.dll (+ jinfer.lib to link against)
#   .\build.ps1 library -Quick     this machine, quick build (-Ob -march=native)
#   .\build.ps1 check              headers vs exports, C example, Python checks
#   .\build.ps1 python             stage the DLL into python\jinfer\lib\windows-x86-64\
#   .\build.ps1 wheel              the wheel, tagged win_amd64 -> target\wheel\
#   .\build.ps1 check-wheel        twine check, a fresh venv with the toknroll wheel, pip install, the checks from outside the tree
param([Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]] $Targets = @('help'), [switch] $Quick)
$Here = $PSScriptRoot; $Name = 'jinfer'; $Opt = '-O3'; $March = 'x86-64-v3'
$Toknroll = "$PSScriptRoot\..\..\toknroll\toknroll-native"
$Headers = 'include\jinfer.h', "$Toknroll\include\toknroll.h"; $Syms = 'jinfer|toknroll'   # the library carries toknroll's C face too
$Example = {
    cl -nologo -std:c11 -W4 -Iinclude "-I$Toknroll\include" examples\c\chat.c -Fe:target\chat.exe -Fo:target\ target\jinfer.lib; if ($LASTEXITCODE) { throw "cl failed" }
    & target\chat.exe; if ($LASTEXITCODE) { throw "chat failed" }
}
$PyTest = { python python\test_jinfer.py; if ($LASTEXITCODE) { throw "checks failed" } }
$WheelDeps = { & "$Toknroll\build.ps1" wheel | Out-Host; (Get-Item "$Toknroll\target\wheel\*.whl").FullName }   # the installed wheel needs toknroll's, built from this tree
$WheelTest = { & wheelenv\Scripts\python ..\python\test_jinfer.py; if ($LASTEXITCODE) { throw "checks failed" } }
. "$PSScriptRoot\..\..\native.ps1"
