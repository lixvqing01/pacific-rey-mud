param(
    [string]$Python = "python",
    [string]$OutputRoot = "outputs",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OriginalLocation = Get-Location
$ExitCode = 0

try {
    Set-Location -LiteralPath $PackageRoot

    $EnvNc = Join-Path $PackageRoot "data\env_gridded.nc"
    $Points = Join-Path $PackageRoot "data\geochem_class_table.parquet"
    $SedRate = Join-Path $PackageRoot "data\Average sedimentation rate.nc"

    foreach ($RequiredFile in @($EnvNc, $Points, $SedRate)) {
        if (-not (Test-Path -LiteralPath $RequiredFile)) {
            throw "Missing required input: $RequiredFile"
        }
    }

    $Arguments = @(
        "-B", "tools/train_fixed.py",
        "--output-root", $OutputRoot,
        "--seeds", "310,411,512,613,714"
    )

    if ($DryRun) {
        $Arguments += "--dry-run"
    }

    & $Python @Arguments
    $ExitCode = $LASTEXITCODE
}
finally {
    Set-Location -LiteralPath $OriginalLocation
}

if ($ExitCode -ne 0) {
    throw "Reproduction command failed with exit code $ExitCode."
}
