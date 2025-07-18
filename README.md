# Simple Autocorrelators - Rydberg Simulation Pipeline

This directory contains a pipeline for running Rydberg lattice simulations using GitHub Actions for parallel computation.

## Overview

The pipeline computes ZZ autocorrelators for Rydberg atoms on a lattice, where each simulation starts with exactly one atom in the excited state (Rydberg state) at a specified site, and all other atoms in the ground state.

## Workflow Structure

```
GitHub Actions Workflow (run-runner.yml)
    ↓
launch_one_batch.py (runs one batch of simulations)
    ↓
runner_ac_data.py (runs individual simulations)
    ↓
Results saved as artifacts
    ↓
fetch.py (downloads and organizes results)
```

## Files Description

- **`run-runner.yml`**: GitHub Actions workflow that parallelizes simulations across multiple runners
- **`launch_one_batch.py`**: Generates bitstrings for a specific batch and calls the runner
- **`runner_ac_data.py`**: Core simulation script that computes ZZ autocorrelators
- **`launch_simulations.py`**: Local alternative to run simulations sequentially
- **`fetch.py`**: Downloads artifacts from GitHub Actions and organizes results

## How to Launch GitHub Actions

1. **Go to your GitHub repository**
2. **Navigate to Actions tab**
3. **Select "Parallel Rydberg Simulations" workflow**
4. **Click "Run workflow"**
5. **Configure parameters:**

### Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `number_of_circuits` | Total number of initial states to simulate | 100 | 256 |
| `batch_size` | How many states each parallel job processes | 10 | 20 |
| `time_to_run` | Simulation time parameter | 1000 | 500 |
| `mid_site` | Index of the initially excited site (0-indexed) | 4 | 4 |
| `Lx` | Lattice width (number of sites in x-direction) | 3 | 3 |
| `Ly` | Lattice height (number of sites in y-direction) | 3 | 3 |

### Parameter Logic

- **Total sites**: `N_sites = Lx × Ly`
- **Valid bitstrings**: Only configurations where `bitstring[mid_site] == '1'`
- **Total valid states**: `2^(N_sites-1)` (since mid_site is fixed to '1')
- **Number of batches**: `ceil(number_of_circuits / batch_size)`

## Example Usage

### From GitHub Web Interface
For a 3×3 lattice with center site excited:
1. Go to your repository → Actions tab
2. Select "Parallel Rydberg Simulations" workflow
3. Click "Run workflow"
4. Set parameters:
   ```
   Lx = 3, Ly = 3 → 9 total sites
   mid_site = 4 → center site (0-indexed)
   number_of_circuits = 100 → process first 100 valid bitstrings
   batch_size = 20 → 5 parallel jobs, each processing 20 states
   ```

### From Terminal (using GitHub CLI)

First, install GitHub CLI if you haven't:
```bash
# macOS
brew install gh

# Login
gh auth login
```

Then trigger the workflow:
```bash
gh workflow run run-runner.yml \
  -f number_of_circuits=100 \
  -f batch_size=20 \
  -f time_to_run=1000 \
  -f mid_site=4 \
  -f Lx=3 \
  -f Ly=3
```

Check workflow status:
```bash
# List recent runs
gh run list --workflow=run-runner.yml

# Watch a specific run
gh run watch <run-id>

# View logs
gh run view <run-id> --log
```

### Alternative: Using curl with GitHub API
```bash
curl -X POST \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/YOUR_USERNAME/YOUR_REPO/actions/workflows/run-runner.yml/dispatches \
  -d '{
    "ref": "main",
    "inputs": {
      "number_of_circuits": "100",
      "batch_size": "20", 
      "time_to_run": "1000",
      "mid_site": "4",
      "Lx": "3",
      "Ly": "3"
    }
  }'
```

## Output Structure

Results are saved as:
```
results/
  L{Lx}_Ly{Ly}_T{time_to_run}/
    correlator_circuit{id}.csv
```

Each CSV contains the real part of the ZZ autocorrelator over time.

## Downloading Results

After the workflow completes:

1. **Run fetch script locally:**
   ```bash
   python fetch.py --hours 24 --outdir results
   ```

2. **Results are organized as:**
   ```
   results/
     L3/
       T1000/
         combined.csv  # All correlators for this configuration
   ```

## Local Development

For testing locally without GitHub Actions:

```bash
# Run a single batch
python simple_autocorrelators/launch_one_batch.py \
  --time 1000 --mid_site 4 --Lx 3 --Ly 3 \
  --batch_size 10 --batch_index 0

# Or run all batches sequentially
python simple_autocorrelators/launch_simulations.py \
  --time 1000 --Lx 3 --Ly 3 --batch_size 50
```

## Checking GitHub Actions Status

### From GitHub Web Interface
1. Go to your repository → **Actions** tab
2. View all workflow runs and their status (✅ completed, 🟡 running, ❌ failed)
3. Click on a specific run to see:
   - Individual job status
   - Real-time logs
   - Artifacts (once completed)

### From Terminal (using GitHub CLI)

**List recent workflow runs:**
```bash
# Show all recent runs
gh run list

# Show runs for specific workflow
gh run list --workflow=run-runner.yml

# Show only running workflows
gh run list --status=in_progress
```

**Monitor a specific run:**
```bash
# Watch run progress in real-time
gh run watch <run-id>

# View run details
gh run view <run-id>

# View logs for a specific run
gh run view <run-id> --log

# View logs for a specific job
gh run view <run-id> --log --job=<job-name>
```

**Check run status programmatically:**
```bash
# Get run status (completed, in_progress, queued, etc.)
gh run view <run-id> --json status --jq '.status'

# Get conclusion (success, failure, cancelled, etc.)
gh run view <run-id> --json conclusion --jq '.conclusion'
```

### Example Monitoring Workflow
```bash
# 1. Start a workflow
gh workflow run run-runner.yml -f number_of_circuits=50

# 2. Get the latest run ID
RUN_ID=$(gh run list --workflow=run-runner.yml --limit=1 --json databaseId --jq '.[0].databaseId')

# 3. Monitor progress
gh run watch $RUN_ID

# 4. Once completed, check if successful
gh run view $RUN_ID --json conclusion --jq '.conclusion'
```

### Status Meanings
- **✅ Success**: All jobs completed successfully, artifacts available
- **🟡 In Progress**: Jobs are currently running
- **⏳ Queued**: Waiting for available runners
- **❌ Failure**: One or more jobs failed (check logs)
- **⚠️ Cancelled**: Manually stopped or timed


## Notes


- **GitHub Actions has usage limits - monitor your quota:**
  - **Check usage via web interface:** Go to your repository → Settings → Billing and plans → Plans and usage
  - **Check via GitHub CLI:** `gh api /user/settings/billing/actions`
  - **Free tier limits:** 2,000 minutes/month for private repos, unlimited for public repos
  - **Monitor during runs:** Each job shows elapsed time in the Actions tab
  - **Estimate costs:** ~1 minute per batch for typical simulations
  
- Larger lattices (higher Lx×Ly) require exponentially more memory
- The `fetch.py` script requires a GitHub token with repo access
- Results are automatically combined and deduplicated by the fetch script


### Usage Monitoring Commands [ if you have billing settings already ]
```bash
# Check current billing cycle usage
gh api /user/settings/billing/actions | jq '.total_minutes_used'

# Check included minutes
gh api /user/settings/billing/actions | jq '.included_minutes'

# Calculate remaining minutes
gh api /user/settings/billing/actions | jq '.included_minutes - .total_minutes_used'
```