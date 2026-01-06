# smoke_test_release.ps1
# Smoke test for release gates
# Usage: .\scripts\smoke_test_release.ps1 -Bundle <bundle> -RunId <runid>

param(
    [Parameter(Mandatory=$true)]
    [string]$Bundle,
    
    [Parameter(Mandatory=$false)]
    [string]$RunId = "dispatch_prop_v2_capfix1",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputRoot = "Output"
)

$ErrorActionPreference = "Stop"

$errors = @()
$warnings = @()

# Helper function to get first existing numeric field from a row (case-insensitive)
function Get-FirstExistingNumericField {
    param(
        [Parameter(Mandatory=$true)]
        [PSCustomObject]$Row,
        
        [Parameter(Mandatory=$true)]
        [string[]]$Candidates
    )
    
    $availableColumns = $Row.PSObject.Properties.Name
    
    foreach ($candidate in $Candidates) {
        # Case-insensitive match
        $matchedColumn = $availableColumns | Where-Object { $_ -eq $candidate -or $_.ToLower() -eq $candidate.ToLower() }
        
        if ($matchedColumn) {
            $value = $Row.$matchedColumn
            if ($null -ne $value -and $value -ne "") {
                try {
                    return [double]$value
                } catch {
                    # Not a valid number, continue to next candidate
                }
            }
        }
    }
    
    # None found - throw helpful error
    $availableColsStr = ($availableColumns | Sort-Object) -join ", "
    throw "None of the candidate fields found: $($Candidates -join ', '). Available columns: $availableColsStr"
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Smoke Test: Release Gates" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bundle: $Bundle" -ForegroundColor Yellow
Write-Host "RunId: $RunId" -ForegroundColor Yellow
Write-Host ""

$bundleDir = "$OutputRoot\runs\$Bundle"

if (-not (Test-Path $bundleDir)) {
    throw "Bundle directory not found: $bundleDir"
}

# Define epochs
$epochs = @("2020", "2025", "2028", "2035_EB", "2035_BB")

# Gate 1: Check existence of all required artefacts
Write-Host "[GATE 1] Checking artefact existence..." -ForegroundColor Green

foreach ($epoch in $epochs) {
    $epochNum = if ($epoch -match "(\d+)") { $matches[1] } else { $epoch }
    
    # Check dispatch summary
    $summaryPath = "$bundleDir\epoch$epoch\dispatch\$RunId\site_dispatch_$epoch`_summary.csv"
    if (-not (Test-Path $summaryPath)) {
        $errors += "Missing dispatch summary: $summaryPath"
    } else {
        Write-Host "  [OK] Found: site_dispatch_$epoch`_summary.csv" -ForegroundColor Gray
    }
    
    # Check incremental electricity (for 2035 epochs)
    if ($epoch -match "2035") {
        $incElecPath = "$bundleDir\epoch$epoch\dispatch\$RunId\signals\incremental_electricity_MW_$epoch.csv"
        if (-not (Test-Path $incElecPath)) {
            $errors += "Missing incremental electricity: $incElecPath"
        } else {
            Write-Host "  [OK] Found: incremental_electricity_MW_$epoch.csv" -ForegroundColor Gray
        }
    }
}

# Check bundle-level artefacts
$kpiPath = "$bundleDir\kpi_table_$RunId.csv"
if (-not (Test-Path $kpiPath)) {
    $warnings += "KPI table not found: $kpiPath (may be generated separately)"
} else {
    Write-Host "  [OK] Found: kpi_table_$RunId.csv" -ForegroundColor Gray
}

$comparePath = "$bundleDir\compare_pathways_2035_EB_vs_BB.csv"
if (-not (Test-Path $comparePath)) {
    $errors += "Missing pathway comparison: $comparePath"
} else {
    Write-Host "  [OK] Found: compare_pathways_2035_EB_vs_BB.csv" -ForegroundColor Gray
}

$rdmEbPath = "$bundleDir\rdm\rdm_summary_2035_EB.csv"
if (-not (Test-Path $rdmEbPath)) {
    $errors += "Missing RDM summary (EB): $rdmEbPath"
} else {
    Write-Host "  [OK] Found: rdm_summary_2035_EB.csv" -ForegroundColor Gray
}

$rdmBbPath = "$bundleDir\rdm\rdm_summary_2035_BB.csv"
if (-not (Test-Path $rdmBbPath)) {
    $errors += "Missing RDM summary (BB): $rdmBbPath"
} else {
    Write-Host "  [OK] Found: rdm_summary_2035_BB.csv" -ForegroundColor Gray
}

$rdmComparePath = "$bundleDir\rdm\rdm_compare_2035_EB_vs_BB.csv"
if (-not (Test-Path $rdmComparePath)) {
    $errors += "Missing RDM comparison: $rdmComparePath"
} else {
    Write-Host "  [OK] Found: rdm_compare_2035_EB_vs_BB.csv" -ForegroundColor Gray
}

Write-Host ""

# Gate 2: Validate TOTAL and UNSERVED rows
Write-Host "[GATE 2] Validating summary rows..." -ForegroundColor Green

foreach ($epoch in $epochs) {
    $summaryPath = "$bundleDir\epoch$epoch\dispatch\$RunId\site_dispatch_$epoch`_summary.csv"
    
    if (Test-Path $summaryPath) {
        $summary = Import-Csv $summaryPath
        
        # Check TOTAL row
        $totalRow = $summary | Where-Object { $_.unit_id -eq "TOTAL" }
        if (-not $totalRow) {
            $errors += "Missing TOTAL row in $summaryPath"
        } else {
            Write-Host "  [OK] TOTAL row exists for $epoch" -ForegroundColor Gray
        }
        
        # Check UNSERVED row
        $unservedRow = $summary | Where-Object { $_.unit_id -eq "UNSERVED" }
        if (-not $unservedRow) {
            $warnings += "Missing UNSERVED row in $summaryPath (may be OK if no unserved energy)"
        } else {
            Write-Host "  [OK] UNSERVED row exists for $epoch" -ForegroundColor Gray
        }
    }
}

Write-Host ""

# Gate 3: Check penalties and unserved are zero (release requirement)
Write-Host "[GATE 3] Checking penalties and unserved (must be zero for release)..." -ForegroundColor Green

foreach ($epoch in $epochs) {
    $summaryPath = "$bundleDir\epoch$epoch\dispatch\$RunId\site_dispatch_$epoch`_summary.csv"
    
    if (Test-Path $summaryPath) {
        $summary = Import-Csv $summaryPath
        
        # Read TOTAL and UNSERVED rows
        $totalRow = $summary | Where-Object { $_.unit_id -eq "TOTAL" }
        $unservedRow = $summary | Where-Object { $_.unit_id -eq "UNSERVED" }
        
        if ($totalRow) {
            try {
                # Get unserved_MWh from first available field (check TOTAL row first, then UNSERVED row)
                $unservedMWh = $null
                $unservedCandidates = @("annual_unserved_MWh", "unserved_MWh", "unserved_MWh")
                
                try {
                    $unservedMWh = Get-FirstExistingNumericField -Row $totalRow -Candidates $unservedCandidates
                } catch {
                    # Try UNSERVED row if available
                    if ($unservedRow) {
                        try {
                            $unservedMWh = Get-FirstExistingNumericField -Row $unservedRow -Candidates $unservedCandidates
                        } catch {
                            throw "Could not find unserved_MWh in TOTAL or UNSERVED row. $_"
                        }
                    } else {
                        throw "Could not find unserved_MWh in TOTAL row and no UNSERVED row exists. $_"
                    }
                }
                
                # Get penalty_cost_nzd from first available field
                $penaltyCostNzd = $null
                $penaltyCandidates = @("annual_penalty_cost_nzd", "total_penalty_cost_nzd", "unserved_penalty_cost_nzd", "unserved_cost_nzd")
                
                try {
                    $penaltyCostNzd = Get-FirstExistingNumericField -Row $totalRow -Candidates $penaltyCandidates
                } catch {
                    # Try UNSERVED row if available
                    if ($unservedRow) {
                        try {
                            $penaltyCostNzd = Get-FirstExistingNumericField -Row $unservedRow -Candidates $penaltyCandidates
                        } catch {
                            throw "Could not find penalty_cost_nzd in TOTAL or UNSERVED row. $_"
                        }
                    } else {
                        throw "Could not find penalty_cost_nzd in TOTAL row and no UNSERVED row exists. $_"
                    }
                }
                
                # Gate condition: unserved_MWh <= 1e-6 AND penalty_cost_nzd <= 1e-6
                $tolerance = 1e-6
                
                if ($unservedMWh -gt $tolerance) {
                    $errors += "Release gate failed: $epoch has unserved energy > tolerance (unserved_MWh = $unservedMWh, tolerance = $tolerance)"
                } else {
                    Write-Host "  [OK] Unserved energy is zero for $epoch (unserved_MWh = $unservedMWh)" -ForegroundColor Gray
                }
                
                if ($penaltyCostNzd -gt $tolerance) {
                    $errors += "Release gate failed: $epoch has penalty cost > tolerance (penalty_cost_nzd = $penaltyCostNzd, tolerance = $tolerance)"
                } else {
                    Write-Host "  [OK] Penalty cost is zero for $epoch (penalty_cost_nzd = $penaltyCostNzd)" -ForegroundColor Gray
                }
            } catch {
                $errors += "Release gate failed: $epoch - $_"
            }
        } else {
            $errors += "Release gate failed: $epoch - TOTAL row not found in summary"
        }
    }
}

Write-Host ""

# Gate 4: Validate 2035_BB has biomass costs and emissions
Write-Host "[GATE 4] Validating 2035_BB biomass costs and emissions..." -ForegroundColor Green

$bbSummaryPath = "$bundleDir\epoch2035_BB\dispatch\$RunId\site_dispatch_2035_BB_summary.csv"
if (Test-Path $bbSummaryPath) {
    $bbSummary = Import-Csv $bbSummaryPath
    
    # Check BB1 unit (biomass boiler)
    $bb1Row = $bbSummary | Where-Object { $_.unit_id -eq "BB1" }
    if ($bb1Row) {
        $fuelCost = [double]$bb1Row.annual_fuel_cost_nzd
        $co2Tonnes = [double]$bb1Row.annual_co2_tonnes
        $carbonCost = [double]$bb1Row.annual_carbon_cost_nzd
        
        if ($fuelCost -le 0) {
            $errors += "Release gate failed: 2035_BB BB1 has annual_fuel_cost_nzd <= 0 (value = $fuelCost)"
        } else {
            Write-Host "  [OK] BB1 annual_fuel_cost_nzd > 0 ($fuelCost)" -ForegroundColor Gray
        }
        
        if ($co2Tonnes -le 0) {
            $warnings += "2035_BB BB1 has annual_co2_tonnes <= 0 (value = $co2Tonnes) - may be OK if biomass is carbon-neutral"
        } else {
            Write-Host "  [OK] BB1 annual_co2_tonnes > 0 ($co2Tonnes)" -ForegroundColor Gray
        }
        
        if ($carbonCost -le 0) {
            $warnings += "2035_BB BB1 has annual_carbon_cost_nzd <= 0 (value = $carbonCost) - may be OK if CO2 is zero"
        } else {
            Write-Host "  [OK] BB1 annual_carbon_cost_nzd > 0 ($carbonCost)" -ForegroundColor Gray
        }
    } else {
        $warnings += "BB1 unit not found in 2035_BB summary (may use different unit naming)"
    }
} else {
    $errors += "Cannot validate 2035_BB: summary file not found"
}

Write-Host ""

# Gate 5: Validate paired futures
Write-Host "[GATE 5] Validating paired futures (EB and BB must use identical future_id sets)..." -ForegroundColor Green

if ((Test-Path $rdmEbPath) -and (Test-Path $rdmBbPath)) {
    $rdmEb = Import-Csv $rdmEbPath
    $rdmBb = Import-Csv $rdmBbPath
    
    $ebFutures = ($rdmEb | Select-Object -ExpandProperty future_id -Unique | Sort-Object)
    $bbFutures = ($rdmBb | Select-Object -ExpandProperty future_id -Unique | Sort-Object)
    
    if ($ebFutures.Count -ne $bbFutures.Count) {
        $errors += "Release gate failed: EB and BB have different numbers of futures (EB: $($ebFutures.Count), BB: $($bbFutures.Count))"
    } else {
        $diff = Compare-Object $ebFutures $bbFutures
        if ($diff) {
            $errors += "Release gate failed: EB and BB have different future_id sets. Differences: $($diff | Out-String)"
        } else {
            Write-Host "  [OK] Paired futures validated: $($ebFutures.Count) futures match between EB and BB" -ForegroundColor Gray
        }
    }
} else {
    $errors += "Cannot validate paired futures: RDM summaries not found"
}

Write-Host ""

# Report results
if ($errors.Count -gt 0) {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "[FAIL] Release gates failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Errors:" -ForegroundColor Red
    foreach ($error in $errors) {
        Write-Host "  - $error" -ForegroundColor Red
    }
    Write-Host ""
    
    if ($warnings.Count -gt 0) {
        Write-Host "Warnings:" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "  - $warning" -ForegroundColor Yellow
        }
    }
    
    throw "Release gates failed. See errors above."
} else {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "[OK] Release gates passed!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    if ($warnings.Count -gt 0) {
        Write-Host ""
        Write-Host "Warnings (non-blocking):" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "  - $warning" -ForegroundColor Yellow
        }
    }
}


