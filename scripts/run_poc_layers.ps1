# run_poc_layers.ps1
# Layer-by-layer runner for PoC pipeline
# Usage: .\scripts\run_poc_layers.ps1 -Bundle <bundle> -RunId <runid> -Epochs "2020,2025,2028,2035_EB,2035_BB" -Layers "all"

param(
    [Parameter(Mandatory=$true)]
    [string]$Bundle,
    
    [Parameter(Mandatory=$false)]
    [string]$RunId = "dispatch_prop_v2_capfix1",
    
    [Parameter(Mandatory=$false)]
    [string]$Epochs = "2020,2025,2028,2035_EB,2035_BB",
    
    [Parameter(Mandatory=$false)]
    [string[]]$Layers = @("all"),
    
    [Parameter(Mandatory=$false)]
    [string]$OutputRoot = "Output"
)

$ErrorActionPreference = "Stop"

# Parse epochs
$epochList = $Epochs -split "," | ForEach-Object { $_.Trim() }

# Parse layers - normalize to handle both string and array inputs
# If $Layers is a single-element array with "all", use all layers
# If $Layers is a single-element array with commas, split it
# If $Layers is already a multi-element array, use it directly
if ($Layers.Count -eq 1 -and $Layers[0] -eq "all") {
    $layerList = @("demandpack", "dispatch", "kpi", "compare", "rdm", "thesis_pack")
} elseif ($Layers.Count -eq 1 -and $Layers[0] -match ",") {
    # Single string containing commas - split it
    $layerList = $Layers[0] -split "," | ForEach-Object { $_.Trim() }
} else {
    # Already an array (or single element without commas) - use directly
    $layerList = $Layers | ForEach-Object { $_.Trim() }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PoC Layer-by-Layer Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bundle: $Bundle" -ForegroundColor Yellow
Write-Host "RunId: $RunId" -ForegroundColor Yellow
Write-Host "Epochs: $($epochList -join ', ')" -ForegroundColor Yellow
Write-Host "Layers: $($layerList -join ', ')" -ForegroundColor Yellow
Write-Host ""

# Get repo root (assume script is in scripts/)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
Set-Location $repoRoot

# Ensure bundle root exists at start
$bundleRoot = Join-Path $OutputRoot "runs\$Bundle"
New-Item -ItemType Directory -Force -Path $bundleRoot | Out-Null
Write-Host "[OK] Bundle root created/verified: $bundleRoot" -ForegroundColor Gray
Write-Host ""

# Layer 1: DemandPack
if ($layerList -contains "demandpack") {
    Write-Host "[LAYER] Running DemandPack and staging to bundle..." -ForegroundColor Green
    
    # Map epoch_tag to eval_year
    # epoch_tag 2020 -> eval_year 2020
    # 2025 -> 2025
    # 2028 -> 2028
    # 2035_EB -> 2035
    # 2035_BB -> 2035
    $epochToEvalYear = @{}
    foreach ($epoch in $epochList) {
        if ($epoch -match "^(\d{4})") {
            $evalYear = [int]$matches[1]
            if (-not $epochToEvalYear.ContainsKey($evalYear)) {
                $epochToEvalYear[$evalYear] = @()
            }
            $epochToEvalYear[$evalYear] += $epoch
        }
    }
    
    # Run DemandPack once per eval_year
    foreach ($evalYear in $epochToEvalYear.Keys | Sort-Object) {
        $epochsForYear = $epochToEvalYear[$evalYear]
        Write-Host "  Running DemandPack for eval_year $evalYear (epochs: $($epochsForYear -join ', '))..." -ForegroundColor Yellow
        
        $configPath = "Input\configs\demandpack_$evalYear.toml"
        
        if (-not (Test-Path $configPath)) {
            Write-Host "    [WARN] Config not found: $configPath, skipping eval_year $evalYear" -ForegroundColor Yellow
            continue
        }
        
        # Generate deterministic run ID (matching Python format: YYYYMMDD_HHMMSS_epoch{epoch}_{config_stem})
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $configFileName = Split-Path -Leaf $configPath
        $configStem = [System.IO.Path]::GetFileNameWithoutExtension($configFileName)
        $dpRunId = "${timestamp}_epoch${evalYear}_${configStem}"
        
        Write-Host "    Using run ID: $dpRunId" -ForegroundColor Gray
        
        # Run DemandPack with explicit run-id
        python -m src.generate_demandpack --config $configPath --epoch $evalYear --run-id $dpRunId
        if ($LASTEXITCODE -ne 0) {
            throw "DemandPack generation failed for eval_year $evalYear"
        }
        
        # Compute run directory deterministically using the same $dpRunId
        $dpRunDir = Join-Path (Join-Path $OutputRoot "runs") $dpRunId
        $dpDemandCsv = Join-Path $dpRunDir ("demandpack\hourly_heat_demand_{0}.csv" -f $evalYear)
        
        # Verify the CSV exists
        if (-not (Test-Path $dpDemandCsv)) {
            throw "Could not find DemandPack output CSV: $dpDemandCsv"
        }
        
        Write-Host "    Found DemandPack output: $dpDemandCsv" -ForegroundColor Gray
        
        # Stage to bundle for each epoch_tag that maps to this eval_year
        foreach ($epochTag in $epochsForYear) {
            $destDir = Join-Path $bundleRoot "epoch$epochTag\demandpack\demandpack"
            $destMetaDir = Join-Path $bundleRoot "epoch$epochTag\demandpack\meta"
            $destFiguresDir = Join-Path $bundleRoot "epoch$epochTag\demandpack\figures"
            
            # Create directories if missing
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
            New-Item -ItemType Directory -Force -Path $destMetaDir | Out-Null
            New-Item -ItemType Directory -Force -Path $destFiguresDir | Out-Null
            
            # Copy demand CSV
            $destCsv = Join-Path $destDir "hourly_heat_demand_$evalYear.csv"
            Copy-Item $dpDemandCsv $destCsv -Force
            Write-Host "    Staged to: epoch$epochTag\demandpack\demandpack\hourly_heat_demand_$evalYear.csv" -ForegroundColor Gray
            
            # Copy manifest for provenance (optional but preferred)
            $sourceManifest = Join-Path $dpRunDir "meta\run_manifest.json"
            if (Test-Path $sourceManifest) {
                $destManifest = Join-Path $destMetaDir "run_manifest.json"
                Copy-Item $sourceManifest $destManifest -Force
                Write-Host "    Staged manifest to: epoch$epochTag\demandpack\meta\run_manifest.json" -ForegroundColor Gray
            }
            
            # Copy DemandPack figures from run directory
            $sourceFiguresDir = Join-Path $dpRunDir "demandpack\figures"
            if (Test-Path $sourceFiguresDir) {
                $sourceFigures = Get-ChildItem -Path $sourceFiguresDir -Filter "demand_$evalYear*.png"
                if ($sourceFigures.Count -gt 0) {
                    foreach ($figure in $sourceFigures) {
                        # Rename to use epoch_tag instead of eval_year
                        $newName = $figure.Name -replace "demand_$evalYear", "demand_$epochTag"
                        Copy-Item $figure.FullName (Join-Path $destFiguresDir $newName) -Force
                        Write-Host "    Staged figure: epoch$epochTag\demandpack\figures\$newName" -ForegroundColor Gray
                    }
                } else {
                    Write-Host "    [WARN] No figures found in $sourceFiguresDir (looking for demand_$evalYear*.png)" -ForegroundColor Yellow
                }
            } else {
                Write-Host "    [WARN] Figures directory not found: $sourceFiguresDir" -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host "[OK] DemandPack layer complete" -ForegroundColor Green
    Write-Host ""
}

# Layer 2: Dispatch
if ($layerList -contains "dispatch") {
    Write-Host "[LAYER] Running site dispatch for epochs..." -ForegroundColor Green
    
    # Ensure bundle exists
    if (-not (Test-Path $bundleRoot)) {
        throw "Bundle directory does not exist: $bundleRoot. Run demandpack layer first."
    }
    
    foreach ($epoch in $epochList) {
        $epochNum = if ($epoch -match "(\d+)") { $matches[1] } else { $epoch }
        
        # Find demand CSV in canonical bundle location
        $demandCsv = "$bundleRoot\epoch$epoch\demandpack\demandpack\hourly_heat_demand_$epochNum.csv"
        
        if (-not (Test-Path $demandCsv)) {
            Write-Host "  [WARN] Demand CSV not found: $demandCsv, skipping $epoch" -ForegroundColor Yellow
            Write-Host "    [INFO] Expected location: $demandCsv" -ForegroundColor Gray
            continue
        }
        
        # Determine utilities CSV
        $utilitiesCsv = $null
        if ($epoch -eq "2035_EB") {
            $utilitiesCsv = "Input\site\utilities\site_utilities_2035_EB.csv"
        } elseif ($epoch -eq "2035_BB") {
            $utilitiesCsv = "Input\site\utilities\site_utilities_2035_BB.csv"
        } else {
            $utilitiesCsv = "Input\site\utilities\site_utilities_$epochNum.csv"
        }
        
        if (-not (Test-Path $utilitiesCsv)) {
            Write-Host "  [WARN] Utilities CSV not found: $utilitiesCsv, skipping $epoch" -ForegroundColor Yellow
            continue
        }
        
        # Build run-id path
        $runIdPath = "$Bundle\epoch$epoch\dispatch\$RunId"
        
        Write-Host "  Running dispatch for $epoch..." -ForegroundColor Yellow
        python -m src.site_dispatch_2020 `
            --epoch $epoch `
            --mode proportional `
            --plot `
            --demand-csv $demandCsv `
            --utilities-csv $utilitiesCsv `
            --output-root $OutputRoot `
            --run-id $runIdPath
        
        if ($LASTEXITCODE -ne 0) {
            throw "Dispatch failed for $epoch"
        }
    }
    Write-Host "[OK] Dispatch layer complete" -ForegroundColor Green
    Write-Host ""
}

# Layer 3: KPI Table
if ($layerList -contains "kpi") {
    Write-Host "[LAYER] Generating KPI table..." -ForegroundColor Green
    Write-Host "  [INFO] KPI table should be generated by KPI export command" -ForegroundColor Yellow
    Write-Host "  [INFO] Expected location: $OutputRoot\runs\$Bundle\kpi_table_$RunId.csv" -ForegroundColor Yellow
    Write-Host "[OK] KPI layer complete (manual step)" -ForegroundColor Green
    Write-Host ""
}

# Layer 4: Compare Pathways
if ($layerList -contains "compare") {
    Write-Host "[LAYER] Comparing 2035 pathways..." -ForegroundColor Green
    
    # Ensure bundle exists
    if (-not (Test-Path $bundleRoot)) {
        throw "Bundle directory does not exist: $bundleRoot. Run dispatch layer first."
    }
    
    $ebRun = "epoch2035_EB\dispatch\$RunId"
    $bbRun = "epoch2035_BB\dispatch\$RunId"
    
    Write-Host "  Comparing EB vs BB..." -ForegroundColor Yellow
    python -m src.compare_pathways_2035 `
        --bundle $Bundle `
        --eb-run $ebRun `
        --bb-run $bbRun `
        --output-root $OutputRoot
    
    if ($LASTEXITCODE -ne 0) {
        throw "Pathway comparison failed"
    }
    Write-Host "[OK] Compare layer complete" -ForegroundColor Green
    Write-Host ""
}

# Layer 5: RDM Screening
if ($layerList -contains "rdm") {
    Write-Host "[LAYER] Running RDM screening for 2035 pathways..." -ForegroundColor Green
    
    # Ensure bundle exists
    if (-not (Test-Path $bundleRoot)) {
        throw "Bundle directory does not exist: $bundleRoot. Run dispatch layer first."
    }
    
    # Run RDM for EB
    Write-Host "  Running RDM for 2035_EB..." -ForegroundColor Yellow
    python -m src.run_rdm_2035 `
        --bundle $Bundle `
        --run-id $RunId `
        --epoch-tag "2035_EB" `
        --output-root $OutputRoot
    
    if ($LASTEXITCODE -ne 0) {
        throw "RDM screening failed for 2035_EB"
    }
    
    # Run RDM for BB
    Write-Host "  Running RDM for 2035_BB..." -ForegroundColor Yellow
    python -m src.run_rdm_2035 `
        --bundle $Bundle `
        --run-id $RunId `
        --epoch-tag "2035_BB" `
        --output-root $OutputRoot
    
    if ($LASTEXITCODE -ne 0) {
        throw "RDM screening failed for 2035_BB"
    }
    
    # Create comparison
    Write-Host "  Creating RDM comparison..." -ForegroundColor Yellow
    python -m src.run_rdm_2035 `
        --bundle $Bundle `
        --run-id $RunId `
        --epoch-tag "2035_EB" `
        --output-root $OutputRoot `
        --create-comparison
    
    if ($LASTEXITCODE -ne 0) {
        throw "RDM comparison failed"
    }
    Write-Host "[OK] RDM layer complete" -ForegroundColor Green
    Write-Host ""
}

# Layer 6: Regional Signals
if ($layerList -contains "regional_signals") {
    Write-Host "[LAYER] Generating regional electricity signals for all epochs..." -ForegroundColor Green
    
    # Ensure bundle exists
    if (-not (Test-Path $bundleRoot)) {
        throw "Bundle directory does not exist: $bundleRoot. Run demandpack layer first."
    }
    
    # Generate signals for each epoch
    foreach ($epoch in $epochList) {
        Write-Host "  Generating regional signals for $epoch..." -ForegroundColor Yellow
        
        # Create output directory
        $regionalSignalsDir = Join-Path $bundleRoot "epoch$epoch\regional_signals\$RunId"
        $signalsSubdir = Join-Path $regionalSignalsDir "signals"
        $figuresSubdir = Join-Path $regionalSignalsDir "figures"
        
        New-Item -ItemType Directory -Force -Path $signalsSubdir | Out-Null
        New-Item -ItemType Directory -Force -Path $figuresSubdir | Out-Null
        
        # Generate signals
        python -m src.generate_regional_signals_poc --epoch-tag $epoch --outdir $regionalSignalsDir
        
        if ($LASTEXITCODE -ne 0) {
            throw "Regional signals generation failed for $epoch"
        }
        
        Write-Host "    [OK] Regional signals generated for $epoch" -ForegroundColor Gray
    }
    
    Write-Host "[OK] Regional signals layer complete" -ForegroundColor Green
    Write-Host ""
}

# Layer 7: Thesis Pack
if ($layerList -contains "thesis_pack") {
    Write-Host "[LAYER] Creating thesis pack..." -ForegroundColor Green
    
    $thesisPackDir = "$OutputRoot\runs\$Bundle\thesis_pack"
    $tablesDir = "$thesisPackDir\tables"
    $figuresDir = "$thesisPackDir\figures"
    
    New-Item -ItemType Directory -Force -Path $tablesDir | Out-Null
    New-Item -ItemType Directory -Force -Path $figuresDir | Out-Null
    
    # Copy key tables
    Write-Host "  Copying tables..." -ForegroundColor Yellow
    
    # KPI table
    $kpiSource = "$OutputRoot\runs\$Bundle\kpi_table_$RunId.csv"
    if (Test-Path $kpiSource) {
        Copy-Item $kpiSource "$tablesDir\kpi_table_$RunId.csv" -Force
        Write-Host "    Copied: kpi_table_$RunId.csv" -ForegroundColor Gray
    } else {
        Write-Host "    [WARN] KPI table not found: $kpiSource (skipping)" -ForegroundColor Yellow
    }
    
    # Pathway comparison
    $compareSource = "$OutputRoot\runs\$Bundle\compare_pathways_2035_EB_vs_BB.csv"
    if (Test-Path $compareSource) {
        Copy-Item $compareSource "$tablesDir\compare_pathways_2035_EB_vs_BB.csv" -Force
        Write-Host "    Copied: compare_pathways_2035_EB_vs_BB.csv" -ForegroundColor Gray
    }
    
    # RDM summaries
    $rdmDir = "$OutputRoot\runs\$Bundle\rdm"
    if (Test-Path "$rdmDir\rdm_summary_2035_EB.csv") {
        Copy-Item "$rdmDir\rdm_summary_2035_EB.csv" "$tablesDir\" -Force
        Write-Host "    Copied: rdm_summary_2035_EB.csv" -ForegroundColor Gray
    }
    if (Test-Path "$rdmDir\rdm_summary_2035_BB.csv") {
        Copy-Item "$rdmDir\rdm_summary_2035_BB.csv" "$tablesDir\" -Force
        Write-Host "    Copied: rdm_summary_2035_BB.csv" -ForegroundColor Gray
    }
    if (Test-Path "$rdmDir\rdm_compare_2035_EB_vs_BB.csv") {
        Copy-Item "$rdmDir\rdm_compare_2035_EB_vs_BB.csv" "$tablesDir\" -Force
        Write-Host "    Copied: rdm_compare_2035_EB_vs_BB.csv" -ForegroundColor Gray
    }
    
    # RDM matrices
    if (Test-Path "$rdmDir\rdm_matrix_2035_EB.csv") {
        Copy-Item "$rdmDir\rdm_matrix_2035_EB.csv" "$tablesDir\" -Force
        Write-Host "    Copied: rdm_matrix_2035_EB.csv" -ForegroundColor Gray
    }
    if (Test-Path "$rdmDir\rdm_matrix_2035_BB.csv") {
        Copy-Item "$rdmDir\rdm_matrix_2035_BB.csv" "$tablesDir\" -Force
        Write-Host "    Copied: rdm_matrix_2035_BB.csv" -ForegroundColor Gray
    }
    
    # Copy dispatch summaries
    foreach ($epoch in $epochList) {
        $summaryFilename = "site_dispatch_$epoch" + "_summary.csv"
        $summarySource = "$OutputRoot\runs\$Bundle\epoch$epoch\dispatch\$RunId\$summaryFilename"
        if (Test-Path $summarySource) {
            Copy-Item $summarySource "$tablesDir\$summaryFilename" -Force
            Write-Host "    Copied: $summaryFilename" -ForegroundColor Gray
        }
    }
    
    # Copy key figures
    Write-Host "  Copying figures..." -ForegroundColor Yellow
    
    # DemandPack figures
    foreach ($epoch in $epochList) {
        $demandpackFiguresDir = "$OutputRoot\runs\$Bundle\epoch$epoch\demandpack\figures"
        if (Test-Path $demandpackFiguresDir) {
            $demandpackFigures = Get-ChildItem -Path $demandpackFiguresDir -Filter "demand_$epoch*.png"
            foreach ($figure in $demandpackFigures) {
                Copy-Item $figure.FullName "$figuresDir\$($figure.Name)" -Force
                Write-Host "    Copied: $($figure.Name)" -ForegroundColor Gray
            }
        }
    }
    
    # Regional signals figures
    foreach ($epoch in $epochList) {
        $regionalSignalsFiguresDir = "$OutputRoot\runs\$Bundle\epoch$epoch\regional_signals\$RunId\figures"
        if (Test-Path $regionalSignalsFiguresDir) {
            $regionalSignalsFigures = Get-ChildItem -Path $regionalSignalsFiguresDir -Filter "*_$epoch.png"
            foreach ($figure in $regionalSignalsFigures) {
                Copy-Item $figure.FullName "$figuresDir\$($figure.Name)" -Force
                Write-Host "    Copied: $($figure.Name)" -ForegroundColor Gray
            }
        }
    }
    
    # Regional signals CSVs (copy to tables)
    Write-Host "  Copying regional signals..." -ForegroundColor Yellow
    foreach ($epoch in $epochList) {
        $regionalSignalsDir = "$OutputRoot\runs\$Bundle\epoch$epoch\regional_signals\$RunId\signals"
        if (Test-Path $regionalSignalsDir) {
            $signalsFiles = Get-ChildItem -Path $regionalSignalsDir -Filter "*_$epoch.csv"
            foreach ($signalFile in $signalsFiles) {
                Copy-Item $signalFile.FullName "$tablesDir\$($signalFile.Name)" -Force
                Write-Host "    Copied: $($signalFile.Name)" -ForegroundColor Gray
            }
        }
    }
    
    # Dispatch stack plots
    foreach ($epoch in $epochList) {
        $figureFilename = "heat_$epoch" + "_unit_stack.png"
        $figureSource = "$OutputRoot\runs\$Bundle\epoch$epoch\dispatch\$RunId\figures\$figureFilename"
        if (Test-Path $figureSource) {
            Copy-Item $figureSource "$figuresDir\$figureFilename" -Force
            Write-Host "    Copied: $figureFilename" -ForegroundColor Gray
        }
    }
    
    # Generate additional thesis figures
    Write-Host "  Generating thesis figures..." -ForegroundColor Yellow
    $epochTagsStr = ($epochList -join ",")
    $thesisFiguresCmd = "python -m src.thesis_figures --bundle $Bundle --run-id $RunId --epoch-tags `"$epochTagsStr`" --output-root $OutputRoot --output-dir `"$figuresDir`" --create-appendix"
    
    try {
        Invoke-Expression $thesisFiguresCmd
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne $null) {
            Write-Host "    [WARN] Thesis figures generation returned exit code $LASTEXITCODE" -ForegroundColor Yellow
        } else {
            Write-Host "    [OK] Thesis figures generated" -ForegroundColor Gray
        }
    } catch {
        Write-Host "    [WARN] Failed to generate thesis figures: $_" -ForegroundColor Yellow
    }
    
    # Create pointers.md - automatically enumerate all files and map to sources
    Write-Host "  Creating pointers.md..." -ForegroundColor Yellow
    
    # Function to map filename to source path
    function Get-SourcePath {
        param(
            [string]$filename,
            [string]$fileType  # "table" or "figure"
        )
        
        # Tables mapping patterns
        if ($fileType -eq "table") {
            # KPI table
            if ($filename -match '^kpi_table_(.+)\.csv$') {
                $runId = $matches[1]
                return "$OutputRoot\runs\$Bundle\kpi_table_$runId.csv"
            }
            
            # Pathway comparison
            if ($filename -eq "compare_pathways_2035_EB_vs_BB.csv") {
                return "$OutputRoot\runs\$Bundle\compare_pathways_2035_EB_vs_BB.csv"
            }
            
            # RDM summaries and matrices
            if ($filename -match '^rdm_(summary|compare|matrix)_(.+)\.csv$') {
                $rdmType = $matches[1]
                $variant = $matches[2]
                return "$OutputRoot\runs\$Bundle\rdm\rdm_$rdmType`_$variant.csv"
            }
            
            # Dispatch summaries
            if ($filename -match '^site_dispatch_(.+)_summary\.csv$') {
                $epoch = $matches[1]
                return "$OutputRoot\runs\$Bundle\epoch$epoch\dispatch\$RunId\site_dispatch_$epoch`_summary.csv"
            }
            
            # Regional signals CSVs
            if ($filename -match '^(headroom_MW|tariff_nzd_per_MWh|gxp_capacity_MW)_(.+)\.csv$') {
                $signalType = $matches[1]
                $epoch = $matches[2]
                return "$OutputRoot\runs\$Bundle\epoch$epoch\regional_signals\$RunId\signals\$signalType`_$epoch.csv"
            }
        }
        
        # Figures mapping patterns
        if ($fileType -eq "figure") {
            # DemandPack figures: demand_<epoch>_*.png
            if ($filename -match '^demand_(.+)_(.+)\.png$') {
                $epoch = $matches[1]
                $suffix = $matches[2]
                return "$OutputRoot\runs\$Bundle\epoch$epoch\demandpack\figures\demand_$epoch`_$suffix.png"
            }
            
            # Regional signals figures: check if filename ends with _<epoch>.png and matches known patterns
            # Patterns: headroom_*, tariff_*, gxp_* (but not heat_* which is dispatch)
            if ($filename -match '^(headroom|tariff|gxp)_.+_(.+)\.png$' -and $filename -notmatch '^heat_') {
                $epoch = $matches[2]
                # Reconstruct full filename (remove .png, add back with epoch)
                $baseName = $filename -replace '\.png$', ''
                return "$OutputRoot\runs\$Bundle\epoch$epoch\regional_signals\$RunId\figures\$baseName.png"
            }
            
            # Dispatch stack plots: heat_<epoch>_unit_stack.png
            if ($filename -match '^heat_(.+)_unit_stack\.png$') {
                $epoch = $matches[1]
                return "$OutputRoot\runs\$Bundle\epoch$epoch\dispatch\$RunId\figures\heat_$epoch`_unit_stack.png"
            }
            
            # 2020 deep-dive figures (generated by thesis_figures.py)
            if ($filename -match '^heat_2020_(timeseries_week|daily_envelope|load_histogram_hourly|weekday_profiles_(Feb|Jun))\.png$') {
                return "$OutputRoot\runs\$Bundle\thesis_pack\figures\appendix\$filename"
            }
            
            # Per-epoch utilisation figures (generated by thesis_figures.py)
            if ($filename -match '^heat_(.+)_(units_online_prop|unit_utilisation_duration_prop)\.png$') {
                $epoch = $matches[1]
                return "$OutputRoot\runs\$Bundle\thesis_pack\figures\$filename"
            }
            
            # RDM comparison figures (generated by thesis_figures.py)
            if ($filename -match '^rdm_2035_compare_(failure_rate_bar|regret_cdf|total_cost_boxplot|shed_fraction_hist)\.png$') {
                return "$OutputRoot\runs\$Bundle\thesis_pack\figures\$filename"
            }
            if ($filename -eq "rdm_2035_compare_regret_cdf_logx.png") {
                return "$OutputRoot\runs\$Bundle\thesis_pack\figures\$filename"
            }
            
            # Tariff/headroom validity figures (generated by thesis_figures.py)
            if ($filename -match '^tariff_(duration_curve|daily_envelope)_(.+)\.png$') {
                $epoch = $matches[2]
                return "$OutputRoot\runs\$Bundle\thesis_pack\figures\$filename"
            }
            if ($filename -match '^headroom_daily_envelope_(.+)\.png$') {
                $epoch = $matches[1]
                return "$OutputRoot\runs\$Bundle\thesis_pack\figures\$filename"
            }
            if ($filename -match '^incremental_electricity_vs_headroom_week_(.+)\.png$') {
                $epoch = $matches[1]
                return "$OutputRoot\runs\$Bundle\thesis_pack\figures\$filename"
            }
        }
        
        # Unknown pattern - return null to indicate we couldn't map it
        return $null
    }
    
    $lines = @()
    $lines += "# Thesis Pack Pointers"
    $lines += ""
    $lines += "This document maps curated thesis outputs to their source paths."
    $lines += ""
    $lines += "## Tables"
    $lines += ""
    
    # Enumerate all tables and map to sources
    if (Test-Path $tablesDir) {
        $tableFiles = Get-ChildItem -Path $tablesDir -File | Sort-Object Name
        foreach ($tableFile in $tableFiles) {
            $filename = $tableFile.Name
            $sourcePath = Get-SourcePath -filename $filename -fileType "table"
            if ($sourcePath) {
                $lines += "- $filename -> $sourcePath"
            } else {
                # Fallback: try to find the file by searching common locations
                $found = $false
                # Try bundle root
                $candidate = "$OutputRoot\runs\$Bundle\$filename"
                if (Test-Path $candidate) {
                    $lines += "- $filename -> $candidate"
                    $found = $true
                }
                # Try RDM directory
                if (-not $found) {
                    $candidate = "$OutputRoot\runs\$Bundle\rdm\$filename"
                    if (Test-Path $candidate) {
                        $lines += "- $filename -> $candidate"
                        $found = $true
                    }
                }
                # Try epoch directories
                if (-not $found) {
                    foreach ($epoch in $epochList) {
                        # Try dispatch
                        $candidate = "$OutputRoot\runs\$Bundle\epoch$epoch\dispatch\$RunId\$filename"
                        if (Test-Path $candidate) {
                            $lines += "- $filename -> $candidate"
                            $found = $true
                            break
                        }
                        # Try regional signals
                        $candidate = "$OutputRoot\runs\$Bundle\epoch$epoch\regional_signals\$RunId\signals\$filename"
                        if (Test-Path $candidate) {
                            $lines += "- $filename -> $candidate"
                            $found = $true
                            break
                        }
                    }
                }
                if (-not $found) {
                    Write-Host "    [WARN] Could not map source for table: $filename" -ForegroundColor Yellow
                    $lines += "- $filename -> [UNKNOWN SOURCE]"
                }
            }
        }
    }
    
    $lines += ""
    $lines += "## Figures"
    $lines += ""
    
    # Enumerate all figures and map to sources
    if (Test-Path $figuresDir) {
        $figureFiles = Get-ChildItem -Path $figuresDir -File | Sort-Object Name
        foreach ($figureFile in $figureFiles) {
            $filename = $figureFile.Name
            $sourcePath = Get-SourcePath -filename $filename -fileType "figure"
            if ($sourcePath) {
                $lines += "- $filename -> $sourcePath"
            } else {
                # Fallback: try to find the file by searching common locations
                $found = $false
                # Try thesis_pack/figures (generated figures)
                $candidate = "$OutputRoot\runs\$Bundle\thesis_pack\figures\$filename"
                if (Test-Path $candidate) {
                    $lines += "- $filename -> $candidate"
                    $found = $true
                }
                # Try thesis_pack/figures/appendix
                if (-not $found) {
                    $candidate = "$OutputRoot\runs\$Bundle\thesis_pack\figures\appendix\$filename"
                    if (Test-Path $candidate) {
                        $lines += "- $filename -> $candidate"
                        $found = $true
                    }
                }
                # Try epoch directories
                if (-not $found) {
                    foreach ($epoch in $epochList) {
                        # Try demandpack
                        $candidate = "$OutputRoot\runs\$Bundle\epoch$epoch\demandpack\figures\$filename"
                        if (Test-Path $candidate) {
                            $lines += "- $filename -> $candidate"
                            $found = $true
                            break
                        }
                        # Try regional signals
                        $candidate = "$OutputRoot\runs\$Bundle\epoch$epoch\regional_signals\$RunId\figures\$filename"
                        if (Test-Path $candidate) {
                            $lines += "- $filename -> $candidate"
                            $found = $true
                            break
                        }
                        # Try dispatch
                        $candidate = "$OutputRoot\runs\$Bundle\epoch$epoch\dispatch\$RunId\figures\$filename"
                        if (Test-Path $candidate) {
                            $lines += "- $filename -> $candidate"
                            $found = $true
                            break
                        }
                    }
                }
                if (-not $found) {
                    Write-Host "    [WARN] Could not map source for figure: $filename" -ForegroundColor Yellow
                    $lines += "- $filename -> [UNKNOWN SOURCE]"
                }
            }
        }
    }
    
    $lines | Set-Content -Path "$thesisPackDir\pointers.md" -Encoding UTF8
    
    Write-Host "[OK] Thesis pack complete" -ForegroundColor Green
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[OK] All layers complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

