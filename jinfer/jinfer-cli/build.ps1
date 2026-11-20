# The jinfer command on Windows: the Makefile's targets, in PowerShell.
#   .\build.ps1 image        what ships: the image at the architecture floor (-march=x86-64-v3) -> target\jinfer.exe
#                            Oracle GraalVM: PGO, from a profile recorded first (pgo-profile) and kept; CE: -O3
#   .\build.ps1 pgo-profile  record the profile again: ..\scripts\pgo.txt's turns on an instrumented image -> target\pgo\
#   .\build.ps1 check        the image answers --version, --help and one instruct turn on the fixture model (-Model)
#   .\build.ps1 dist         the release zip, a prefix tree: bin\jinfer.exe, README, LICENSE, NOTICE -> target\dist\
#   .\build.ps1 check-dist   unpack the zip and run its jinfer as check does
param([Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]] $Targets = @('help'), [string] $Model = 'LiquidAI/LFM2.5-350M-GGUF:Q8_0')
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\scripts\toolchain.ps1"
Push-Location $PSScriptRoot
function Build-Image([string] $Opt) {   # no clean: the image is linked from the modules' classes, not the shaded jar
    Enter-Toolchain
    Push-Location $Root
    # quoted: PowerShell 5.1 splits an unquoted -argument at its first dot; $null passes no argument, so the pom's -O3 stays
    try { mvn -Pnative -pl jinfer/jinfer-cli -am package -DskipTests "-Djinfer.image.march=x86-64-v3" $Opt; if ($LASTEXITCODE) { throw "mvn failed" } }
    finally { Pop-Location }
}
# Oracle GraalVM builds what ships optimized for a profile its instrumented image records first (PGO), kept for
# libjinfer's build and the next image's; GraalVM CE builds at the pom's -O3
function Build-Shipped {
    Enter-Toolchain
    if (-not ((native-image --version | Select-Object -Index 1) -match 'Oracle GraalVM')) { Build-Image; return }
    if (-not (Test-Path target\pgo\profiles)) { Record-Profile }
    Build-Image "-Djinfer.image.opt=--pgo=$(Get-Content target\pgo\profiles)"
}
# The profile, as the Makefile's: each line of ..\scripts\pgo.txt, a turn's arguments, on the instrumented image, a
# process and a profile each, a prompt and 4 tokens -> target\pgo\, listed in target\pgo\profiles
function Record-Profile {
    Build-Image '-Djinfer.image.opt=--pgo-instrument'
    Remove-Item -Recurse -Force target\pgo -ErrorAction Ignore; New-Item -ItemType Directory -Force target\pgo | Out-Null
    Move-Item target\jinfer.exe target\pgo\train.exe
    $prompt = 'In two sentences, explain why a compiler that sees how a program actually runs can make it faster than one that only reads its source code.'
    $i = 0
    foreach ($turn in Get-Content ..\scripts\pgo.txt | Where-Object { $_ -notmatch '^(#|\s*$)' }) {
        $i++; "pgo ${i}: $turn"
        $arguments = -split $turn   # array elements reach the image as they are, -D's dots unsplit
        & target\pgo\train.exe "-XX:ProfilesDumpFile=$PSScriptRoot\target\pgo\$i.iprof" @arguments -n 4 instruct $prompt | Out-Null
        if ($LASTEXITCODE) { throw "pgo turn $i failed" }
    }
    [IO.File]::WriteAllText("$PSScriptRoot\target\pgo\profiles", ((Get-ChildItem target\pgo\*.iprof).FullName -join ','))
}
function Run-Check([string] $Bin = 'target\jinfer.exe') {
    & $Bin --version; if ($LASTEXITCODE) { throw "--version failed" }
    & $Bin --help | Out-Null; if ($LASTEXITCODE) { throw "--help failed" }
    & $Bin -m $Model --temp 0 instruct "Reply with the single word ready." | Out-Null; if ($LASTEXITCODE) { throw "instruct failed" }
}
function Dist-Name { 'jinfer-' + ((& target\jinfer.exe --version) -split ' ')[1] + '-windows-x86-64' }   # the image's own word
function Build-Dist {
    $dist = Dist-Name
    Remove-Item -Recurse -Force "target\dist\$dist", "target\dist\$dist.zip" -ErrorAction Ignore
    New-Item -ItemType Directory -Force "target\dist\$dist\bin" | Out-Null
    Copy-Item target\jinfer.exe "target\dist\$dist\bin\"; Copy-Item README.md, ..\jinfer-native\NOTICE, "$Root\LICENSE" "target\dist\$dist\"   # the notice: the same image, the same terms
    Copy-VcRuntime target\jinfer.exe "target\dist\$dist\bin\"
    New-Zip "$PSScriptRoot\target\dist" $dist
    $hash = (Get-FileHash "target\dist\$dist.zip" -Algorithm SHA256).Hash.ToLower()
    [IO.File]::WriteAllText("$PSScriptRoot\target\dist\$dist.zip.sha256", "$hash  $dist.zip`n")
    Get-Item "target\dist\$dist.zip" | Format-Table Name, Length | Out-String | Write-Host; Get-Content "target\dist\$dist.zip.sha256"
}
function Check-Dist {
    $dist = Dist-Name
    if (Test-Path target\distenv) { Remove-Item -Recurse -Force target\distenv }
    Expand-Archive "target\dist\$dist.zip" target\distenv
    Run-Check "target\distenv\$dist\bin\jinfer.exe"
    "the unpacked $dist answers --version, --help and an instruct turn"
}
try {
    foreach ($t in $Targets) {
        switch ($t) {
            'help'       { Get-Content "$PSScriptRoot\build.ps1" | Select-Object -Skip 1 | Select-Object -First 6 | ForEach-Object { $_ -replace '^#\s?', '' } }
            'image'      { Build-Shipped }
            'pgo-profile' { Record-Profile }
            'check'      { Run-Check }
            'dist'       { Build-Dist }
            'check-dist' { Check-Dist }
            default      { throw "no target '$t'" }
        }
    }
} finally { Pop-Location }
