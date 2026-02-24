# Initialize DVC with a local remote so pull_data and commit_and_push tasks can run (e.g. in Docker).
# Run once from project root: .\scripts\setup_dvc_local.ps1
#
# If you get: cannot import name '_DIR_MARK' from 'pathspec.patterns.gitwildmatch'
# run first: pip install "pathspec>=0.11,<0.12"
# then run this script again.

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DvcRemote = Join-Path $ProjectRoot "data" "dvc_remote"

Set-Location $ProjectRoot

if (-not (Test-Path ".dvc")) {
    Write-Host "Running dvc init..."
    dvc init
} else {
    Write-Host "DVC already initialized."
}

New-Item -ItemType Directory -Path $DvcRemote -Force | Out-Null
Write-Host "Local remote dir: $DvcRemote"

# Use relative path so the same config works in Docker (mount at /opt/airflow) and on Windows
$RemoteExists = dvc remote list 2>$null | Select-String -Pattern "local"
if (-not $RemoteExists) {
    dvc remote add -d local "data/dvc_remote"
    Write-Host "Added default remote 'local' -> data/dvc_remote"
} else {
    Write-Host "Remote 'local' already exists."
}

Write-Host "Done. You can run the DAG; pull_data and commit_and_push will use this local store."
