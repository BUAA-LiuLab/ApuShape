param(
    [string]$Exe = "",
    [string[]]$Images = @(),
    [string]$Model = "",
    [string]$Output = "",
    [double]$Confidence = 0.4,
    [int]$MinTotalContours = 0
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $Output) {
    $Output = Join-Path $Root "results"
}

if (-not $Exe) {
    $expectedExe = Join-Path $Root "bin\ShapeStarSmoke\ShapeStarSmoke.exe"
    if (Test-Path $expectedExe) {
        $Exe = $expectedExe
    } else {
        $foundExe = Get-ChildItem -LiteralPath (Join-Path $Root "bin") -Recurse -Filter "ShapeStarSmoke.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($foundExe) {
            $Exe = $foundExe.FullName
        }
    }
}

if (-not $Exe -or -not (Test-Path $Exe)) {
    throw "ShapeStarSmoke.exe was not found under $Root\bin"
}

if (-not $Model) {
    $foundModel = Get-ChildItem -LiteralPath (Join-Path $Root "models") -Filter "*.pth" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($foundModel) {
        $Model = $foundModel.FullName
    }
}

if (-not $Model -or -not (Test-Path $Model)) {
    throw "Model checkpoint (.pth) was not found under $Root\models"
}

if (-not $Images -or $Images.Count -eq 0) {
    $Images = Get-ChildItem -LiteralPath (Join-Path $Root "images") -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".tif", ".tiff", ".png", ".jpg", ".jpeg") } |
        Sort-Object Name |
        ForEach-Object { $_.FullName }
}

if (-not $Images -or $Images.Count -eq 0) {
    throw "No test images were found under $Root\images"
}

if (Test-Path $Output) {
    Remove-Item -LiteralPath $Output -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Output | Out-Null

Write-Host "ShapeStarSmoke executable: $Exe"
Write-Host "ShapeStar model: $Model"
Write-Host "Input images:"
foreach ($image in $Images) {
    Write-Host " - $image"
}
Write-Host "Output directory: $Output"
Write-Host "Confidence threshold: $Confidence"

& $Exe -i $Images -m $Model -o $Output --device cpu --confidence $Confidence --min-contours 0
$code = $LASTEXITCODE
if ($code -ne 0) {
    throw "ShapeStarSmoke failed with exit code $code"
}

$totalContours = 0
foreach ($image in $Images) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($image)
    $jsonPath = Join-Path $Output "$stem.json"
    $maskPath = Join-Path $Output "$stem.png"
    $overlayPath = Join-Path $Output "${stem}_overlay.png"

    foreach ($path in @($jsonPath, $maskPath, $overlayPath)) {
        if (-not (Test-Path $path)) {
            throw "Missing expected output: $path"
        }
    }

    $json = Get-Content $jsonPath -Raw | ConvertFrom-Json
    $count = [int]$json.contour_count
    $totalContours += $count
    Write-Host "Verified ${stem}: contour_count=$count"
    Write-Host "Generated: $jsonPath"
    Write-Host "Generated: $maskPath"
    Write-Host "Generated: $overlayPath"
}

if ($totalContours -lt $MinTotalContours) {
    throw "Smoke test produced $totalContours total contours, expected at least $MinTotalContours."
}

Write-Host "ShapeStar smoke test completed: image_count=$($Images.Count), total_contours=$totalContours"
