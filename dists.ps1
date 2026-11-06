# make dists on Windows, where make is not native: this machine's three release archives, the two libraries for C and
# the jinfer command, built and checked
# from a clean export of a ref (a tag, a commit, any tree-ish) -> target\dists\. Upload them to the release with
# gh release upload <tag> target\dists\*.zip target\dists\*.sha256.
#   .\dists.ps1 [-Ref <tag>]
param([string] $Ref = 'HEAD')
$ErrorActionPreference = 'Stop'
$export = "$PSScriptRoot\target\dists-src"
$dists = "$PSScriptRoot\target\dists"
foreach ($stale in $export, "$export.tar") { if (Test-Path $stale) { Remove-Item -Recurse -Force $stale } }
New-Item -ItemType Directory -Force $export, $dists | Out-Null
git -C $PSScriptRoot -c core.autocrlf=false archive -o "$export.tar" $Ref; if ($LASTEXITCODE) { throw "git archive failed" }   # the files as committed
& "$env:SystemRoot\System32\tar.exe" -xf "$export.tar" -C $export; if ($LASTEXITCODE) { throw "tar failed" }   # Windows' own tar
& "$export\toknroll\toknroll-native\build.ps1" library check dist
& "$export\jinfer\jinfer-native\build.ps1" library check dist
& "$export\jinfer\jinfer-cli\build.ps1" image check dist check-dist
foreach ($z in Get-Item "$export\toknroll\toknroll-native\target\dist\*.zip", "$export\jinfer\jinfer-native\target\dist\*.zip", "$export\jinfer\jinfer-cli\target\dist\*.zip") {
    $lib = $z.Name.Split('-')[0]   # a new archive replaces the library's or command's archive for this platform
    Remove-Item "$dists\$lib-*-windows-x86-64.zip", "$dists\$lib-*-windows-x86-64.zip.sha256" -ErrorAction Ignore
    Copy-Item $z, "$($z.FullName).sha256" $dists
}
Get-Item "$dists\*.zip" | ForEach-Object { '{0:N1} MB {1}' -f ($_.Length / 1e6), $_.Name }
