# ============================================================
# Marine Debris Detection & Drift Prediction - Run Pipeline
# ============================================================
# Usage:
#   .\run.ps1                  # Full pipeline (train + eval + postprocess + drift)
#   .\run.ps1 -SkipTrain       # Skip training, eval existing checkpoint
#   .\run.ps1 -Resume          # Resume interrupted training
#   .\run.ps1 -WithRF          # Also train Random Forest
#   .\run.ps1 -EvalOnly        # Only evaluate + postprocess
#   .\run.ps1 -TrainOnly       # Only train (no eval/postprocess/drift)
# ============================================================

param(
    [switch]$SkipTrain,
    [switch]$Resume,
    [switch]$WithRF,
    [switch]$WithSI,
    [switch]$WithGLCM,
    [switch]$EvalOnly,
    [switch]$TrainOnly,
    [switch]$NoTTA,
    [int]$Epochs = 120,
    [int]$Batch = 16,
    [double]$LR = 1e-4,
    [int]$Patience = 30,
    [double]$WeightDecay = 5e-5,
    [int]$Workers = 4,
    [string]$Checkpoint = "",
    [string]$OceanNC = "",
    [string]$WindNC = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$ProjectRoot = (Resolve-Path "$PSScriptRoot/..").Path
$OutDir = Join-Path $ProjectRoot "outputs"
$CkptDir = Join-Path $ProjectRoot "checkpoints"

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Marine Debris Detection Pipeline" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Epochs: $Epochs  Batch: $Batch  LR: $LR" -ForegroundColor DarkCyan
Write-Host "  Patience: $Patience  Workers: $Workers" -ForegroundColor DarkCyan
Write-Host ""

# --- Step 1: Train ---
if (-not $SkipTrain -and -not $EvalOnly) {
    Write-Host "[1/4] Training model..." -ForegroundColor Yellow
    $trainArgs = @(
        "train.py",
        "--epochs", $Epochs,
        "--batch", $Batch,
        "--lr", $LR,
        "--patience", $Patience,
        "--weight_decay", $WeightDecay,
        "--workers", $Workers
    )
    if ($Resume) { $trainArgs += "--resume" }
    python @trainArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "Training failed!" -ForegroundColor Red; exit 1 }
    Write-Host "[1/4] Training complete." -ForegroundColor Green
} else {
    Write-Host "[1/4] Skipping training." -ForegroundColor DarkGray
}

if ($TrainOnly) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " Training complete! (TrainOnly mode)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    exit 0
}

# --- Step 2: Evaluate ---
Write-Host "[2/4] Evaluating on test set..." -ForegroundColor Yellow
$evalArgs = @("evaluate.py", "--split", "test", "--save_masks", "--threshold", "0.5")
if (-not $NoTTA) { $evalArgs += "--tta" }
if ($Checkpoint -ne "") { $evalArgs += @("--ckpt", $Checkpoint) }
python @evalArgs
if ($LASTEXITCODE -ne 0) { Write-Host "Evaluation failed!" -ForegroundColor Red; exit 1 }
Write-Host "[2/4] Evaluation complete." -ForegroundColor Green

# --- Step 3: Post-process ---
Write-Host "[3/4] Post-processing predictions..." -ForegroundColor Yellow
python postprocess.py --pred_dir "$OutDir/predicted_test" --out_dir "$OutDir/geospatial"
if ($LASTEXITCODE -ne 0) { Write-Host "Post-processing failed!" -ForegroundColor Red; exit 1 }
Write-Host "[3/4] Post-processing complete." -ForegroundColor Green

# --- Step 4: Drift prediction ---
Write-Host "[4/4] Running drift prediction..." -ForegroundColor Yellow
$driftArgs = @("drift_prediction/drift.py", "--geojson", "$OutDir/geospatial/all_debris.geojson", "--out_dir", "$OutDir/drift")
if ($OceanNC -ne "") { $driftArgs += @("--ocean_nc", $OceanNC) }
if ($WindNC -ne "")  { $driftArgs += @("--wind_nc", $WindNC) }
python @driftArgs
if ($LASTEXITCODE -ne 0) { Write-Host "Drift prediction failed!" -ForegroundColor Red; exit 1 }
Write-Host "[4/4] Drift prediction complete." -ForegroundColor Green

# --- Optional: Random Forest ---
if ($WithRF) {
    Write-Host "[RF] Training Random Forest..." -ForegroundColor Yellow
    $rfArgs = @("random_forest/train_eval.py")
    if ($WithSI)   { $rfArgs += "--use_si" }
    if ($WithGLCM) { $rfArgs += "--use_glcm" }
    python @rfArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "RF training failed!" -ForegroundColor Red; exit 1 }
    Write-Host "[RF] Random Forest complete." -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Pipeline complete!" -ForegroundColor Green
Write-Host " Outputs -> $OutDir" -ForegroundColor Green
Write-Host " Checkpoints -> $CkptDir" -ForegroundColor Green
Write-Host " Logs -> $(Join-Path $ProjectRoot 'logs')" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
