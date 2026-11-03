param(
    [string]$Python = "python",
    [string]$OutputDir = "outputs/pretrained_prediction",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OriginalLocation = Get-Location
$ExitCode = 0

try {
    Set-Location -LiteralPath $PackageRoot
    $Arguments = @(
        "-B", "-m", "scripts.predict_rey_continuous_grid",
        "--model-run-dirs", "models/seed714",
        "--env-nc", "data/env_gridded.nc",
        "--sed-rate-nc", "data/Average sedimentation rate.nc",
        "--output-dir", $OutputDir,
        "--batch-size", "16384",
        "--age-batch-size", "2",
        "--num-workers", "0",
        "--prefetch-factor", "4",
        "--patch-mode", "fast_grid"
    )

    if ($Quick) {
        $Arguments += @("--age-layers-ka", "0", "--target-resolution-deg", "2.0", "--stride", "20")
    }
    else {
        $Arguments += @(
            "--age-layers-ka", "0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2580",
            "--target-resolution-deg", "0.1",
            "--stride", "1"
        )
    }

    & $Python @Arguments
    $ExitCode = $LASTEXITCODE
}
finally {
    Set-Location -LiteralPath $OriginalLocation
}

if ($ExitCode -ne 0) {
    throw "Pretrained prediction command failed with exit code $ExitCode."
}
