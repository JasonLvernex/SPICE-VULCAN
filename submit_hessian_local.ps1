<#
.SYNOPSIS
    Runs Uncert_01_Laplacian_Covariance.py (voxel-wise Hessian) for a sweep of
    lambda values, each split across several parallel Python processes on a
    single multi-core Windows workstation.

.DESCRIPTION
    Windows caps a single ProcessPoolExecutor at 61 workers (WaitForMultipleObjects
    hard limit), regardless of how many physical cores the machine has. To use more
    of a big workstation (e.g. 128 cores), this script launches -NJobs separate
    Python processes as PowerShell background jobs per lambda, each with its own
    <=61-worker pool, each handling a slice of the brain-voxel list via
    --vox-start/--vox-end. Lambdas are run one at a time (each lambda's jobs use
    the full core budget); voxel jobs within a lambda run in parallel.

    Run-tags are derived as "w<Wmax>_l<lambda>" using Python's own :g formatting
    (via a tiny python -c call) so they exactly match the directory names
    recon_01_run_spice.py actually created - e.g. lambda=1e-4 becomes
    "w5000_l0.0001", not "w5000_l1e-4". Hand-formatting this in PowerShell would
    silently produce mismatched paths for some lambdas (verified: 1e-4, 1e-3,
    3e-3, 4e-3, 1e-2, 1e-1 all fall on the .NET/Python formatting divergence).

    The brain voxel count (used to size each job's --vox-start/--vox-end slice)
    is auto-detected by calling make_brain_mask() with the same threshold/
    erosion/cleanup settings passed to Uncert_01 itself (unless -TotalVoxels is
    given explicitly). Overshooting this (e.g. guessing the full 64x64=4096
    grid when only ~1700 voxels are actually in-brain) does NOT "safely clip" -
    it unbalances the split: with a large enough overshoot, an early job's
    slice can absorb the entire real voxel list while later jobs get an empty
    slice and do nothing.

    Run from the repo root (or anywhere - it cd's to its own location first).

.EXAMPLE
    .\submit_hessian_local.ps1
    .\submit_hessian_local.ps1 -Subject invivo_260623_02 -Lambdas 1e-4,1e-3 -NJobs 3
#>

param(
    [string]  $Subject          = "invivo_260717_01",
    [string]  $BasisDir         = "./basis_shifted/",
    [string]  $Wmax             = "5000",
    [string[]]$Lambdas          = @("1e-7","1e-6","6e-6","1e-5","4e-5","1e-4","1e-3","3e-3","4e-3","1e-2","1e-1"),
    [int]     $Rank             = 10,
    [double]  $BrainThreshold   = 0.10,
    [int]     $BrainErosion     = 1,
    [switch]  $NoBrainMaskCleanup,      # pass this to skip --brain-mask-cleanup
    [int]     $CgMaxiter        = 300,
    [double]  $CgRtol           = 1e-3,
    [int]     $TotalVoxels      = 0,    # 0 = auto-detect actual brain voxel count; >0 overrides
    [int]     $NJobs            = 2,    # parallel python processes per lambda
    [int]     $MaxWorkersPerJob = 61,   # Windows hard cap per ProcessPoolExecutor
    [string]  $PythonExe        = "d:\anaconda3\envs\VULCAN\python.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
Set-Location $repoRoot

function Get-RunTag([string]$wmaxStr, [string]$lamStr) {
    # Mirror recon_01_run_spice.py's f"w{wmax:g}_l{lambda1:g}" exactly. wmax/lambda
    # are passed through as raw strings (sys.argv) and parsed by Python itself, so
    # no PowerShell double->string round-trip can perturb the value beforehand.
    (& $PythonExe -c "import sys; print(f'w{float(sys.argv[1]):g}_l{float(sys.argv[2]):g}')" $wmaxStr $lamStr).Trim()
}

function Get-BrainVoxelCount([string]$dataDir, [double]$threshold, [int]$erosion, [bool]$cleanup) {
    # Same make_brain_mask() call Uncert_01 itself uses, so the count (and hence
    # the chunk split) exactly matches what the script will actually process -
    # no guessing/overshooting the grid size and hoping slicing clips evenly.
    $cleanupArg = if ($cleanup) { "1" } else { "0" }
    $out = & $PythonExe -c "import sys, numpy as np; sys.path.insert(0, '.'); from utils.pipeline_utils import make_brain_mask; wref = np.load(sys.argv[1] + '/wref_o.npy', mmap_mode='r'); _, mask, _ = make_brain_mask(wref, float(sys.argv[2]), int(sys.argv[3]), cleanup=bool(int(sys.argv[4]))); print(int(mask.sum()))" $dataDir $threshold $erosion $cleanupArg
    return [int]($out.Trim())
}

if ($TotalVoxels -le 0) {
    $cleanupOn = -not $NoBrainMaskCleanup
    Write-Host "[hessian] auto-detecting brain voxel count for $Subject (threshold=$BrainThreshold erosion=$BrainErosion cleanup=$cleanupOn) ..."
    $TotalVoxels = Get-BrainVoxelCount -dataDir "data/processed/$Subject" -threshold $BrainThreshold -erosion $BrainErosion -cleanup $cleanupOn
    Write-Host "[hessian] detected $TotalVoxels brain voxels"
}

$chunk = [Math]::Ceiling($TotalVoxels / $NJobs)
Write-Host "[hessian] subject=$Subject  wmax=$Wmax  lambdas=$($Lambdas -join ', ')"
Write-Host "[hessian] per lambda: $NJobs job(s) x $MaxWorkersPerJob workers, chunk size=$chunk"

foreach ($lam in $Lambdas) {
    $runTag = Get-RunTag -wmaxStr $Wmax -lamStr $lam
    $hessDir = "output/$Subject/hessian_$runTag"

    Write-Host ""
    Write-Host "=============================================================="
    Write-Host "[hessian] lambda=$lam  ->  run-tag=$runTag"
    Write-Host "=============================================================="

    $commonArgs = @(
        "--data-dir",        "data/processed/$Subject",
        "--basis-dir",        $BasisDir,
        "--run-tag",          $runTag,
        "--rank",             "$Rank",
        "--brain-threshold",  "$BrainThreshold",
        "--brain-erosion",    "$BrainErosion",
        "--cg-maxiter",       "$CgMaxiter",
        "--cg-rtol",          "$CgRtol",
        "--max-workers",      "$MaxWorkersPerJob"
    )
    if (-not $NoBrainMaskCleanup) { $commonArgs += "--brain-mask-cleanup" }

    $jobs = @()
    for ($i = 0; $i -lt $NJobs; $i++) {
        $voxStart = $i * $chunk
        $voxEnd   = [Math]::Min($voxStart + $chunk, $TotalVoxels)
        Write-Host "[hessian]   job $i : voxels [$voxStart : $voxEnd)"

        $jobs += Start-Job -Name "hess_${runTag}_$i" -ScriptBlock {
            Set-Location $using:repoRoot
            $env:KMP_DUPLICATE_LIB_OK = "TRUE"
            # Python fully buffers stdout when it's not a real terminal (i.e. always,
            # when run under a job) -- most of the setup-phase prints in Uncert_01
            # don't pass flush=True, so without this nothing shows up until the
            # buffer fills or the process exits. This forces every print through
            # immediately, in the main process AND in each worker it forks.
            $env:PYTHONUNBUFFERED = "1"
            & $using:PythonExe scripts/uncertainty/analytical/Uncert_01_Laplacian_Covariance.py `
                @using:commonArgs --vox-start $using:voxStart --vox-end $using:voxEnd
        }
    }

    Write-Host "[hessian]   $($jobs.Count) job(s) running - streaming output every 5s (Ctrl+C to stop watching, jobs keep running) ..."
    while ($jobs | Where-Object { $_.State -eq 'Running' }) {
        Start-Sleep -Seconds 5
        foreach ($j in $jobs) {
            $jobErr = $null
            $out = Receive-Job $j -ErrorVariable jobErr -ErrorAction SilentlyContinue
            foreach ($line in $out) { Write-Host "[$($j.Name)] $line" }
            # tqdm writes its progress bar to stderr; PowerShell wraps stderr lines
            # from a job as ErrorRecords, so surface them without treating them as
            # fatal (ErrorActionPreference=Stop would otherwise kill this script on
            # the first progress-bar tick).
            foreach ($e in $jobErr) { Write-Host "[$($j.Name)] [stderr] $e" }
        }
    }
    # drain anything that landed between the last poll and completion
    foreach ($j in $jobs) {
        $jobErr = $null
        $out = Receive-Job $j -ErrorVariable jobErr -ErrorAction SilentlyContinue
        foreach ($line in $out) { Write-Host "[$($j.Name)] $line" }
        foreach ($e in $jobErr) { Write-Host "[$($j.Name)] [stderr] $e" }
        Write-Host "[$($j.Name)] state: $($j.State)"
    }
    $jobs | Remove-Job

    Write-Host "[hessian] lambda=$lam done  ->  $hessDir/"
}

Write-Host ""
Write-Host "[hessian] Sweep complete: $($Lambdas.Count) lambda(s) x $NJobs job(s) each."
