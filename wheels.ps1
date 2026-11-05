# make wheels and make publish-wheels on Windows, where make is not native: this machine's two wheels, built and
# checked from a clean export of a ref (a tag, a commit, any tree-ish) -> target\wheels\.
#   .\wheels.ps1 [-Ref <tag>]            build them
#   .\wheels.ps1 -Publish [-Ref <tag>]   upload target\wheels\ to PyPI after twine check, skipping wheels it
#                                        holds; each must be built from the ref
param([string] $Ref = 'HEAD', [switch] $Publish)
$ErrorActionPreference = 'Stop'
$export = "$PSScriptRoot\target\wheels-src"
$wheels = "$PSScriptRoot\target\wheels"
if ($Publish) {   # from the release tools' venv, native.ps1's by the same path
    $tools = "$PSScriptRoot\target\pytools\" + (python -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep=chr(45))')
    if (-not (Test-Path "$tools\Scripts\twine.exe")) {
        python -m venv $tools; if ($LASTEXITCODE) { throw "venv failed" }
        & "$tools\Scripts\pip" install -q twine; if ($LASTEXITCODE) { throw "pip install twine failed" }
    }
    $commit = git -C $PSScriptRoot rev-parse "$Ref^{commit}"; if ($LASTEXITCODE) { throw "no commit $Ref" }
    foreach ($w in Get-ChildItem "$wheels\*.whl") {   # git archive wrote its commit into the package's "built from"
        $init = & "$env:SystemRoot\System32\tar.exe" -xOf $w.FullName "$($w.Name.Split('-')[0])/__init__.py"
        if (-not ($init -match "built from $commit")) { throw "$($w.Name) is not built from $Ref, $commit" }
    }
    & "$tools\Scripts\twine" check --strict "$wheels\*.whl"; if ($LASTEXITCODE) { throw "twine check failed" }
    & "$tools\Scripts\twine" upload --skip-existing "$wheels\*.whl"; if ($LASTEXITCODE) { throw "twine upload failed" }
    return
}
foreach ($stale in $export, "$export.tar") { if (Test-Path $stale) { Remove-Item -Recurse -Force $stale } }
New-Item -ItemType Directory -Force $export, $wheels | Out-Null
# the files as committed: Git for Windows would hand the archive CRLF line ends
git -C $PSScriptRoot -c core.autocrlf=false archive -o "$export.tar" $Ref; if ($LASTEXITCODE) { throw "git archive failed" }
# Windows' own tar: a Git usr\bin on the PATH brings GNU tar, which reads C:\ as a remote host
& "$env:SystemRoot\System32\tar.exe" -xf "$export.tar" -C $export; if ($LASTEXITCODE) { throw "tar failed" }
& "$export\toknroll\toknroll-native\build.ps1" library check-wheel
& "$export\jinfer\jinfer-native\build.ps1" library check-wheel
foreach ($w in Get-Item "$export\toknroll\toknroll-native\target\wheel\*.whl", "$export\jinfer\jinfer-native\target\wheel\*.whl") {
    $parts = $w.Name.Split('-')   # a new wheel replaces the package's wheel for its platform
    Remove-Item "$wheels\$($parts[0])-*-$($parts[-1])" -ErrorAction Ignore
    Copy-Item $w $wheels
}
Get-Item "$wheels\*.whl" | ForEach-Object { '{0:N1} MB {1}' -f ($_.Length / 1e6), $_.Name }
