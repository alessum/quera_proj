#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import shutil
import zipfile
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from itertools import groupby

GITHUB_API = "https://api.github.com"

def request_with_retries(url, headers=None, params=None, stream=False,
                         timeout=10, max_retries=3, backoff=2):
    headers = headers or {}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                url, headers=headers, params=params,
                stream=stream, timeout=timeout
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == max_retries:
                raise
            time.sleep(backoff)

def get_workflow_runs(owner, repo, workflow_filename, token, since_dt):
    """
    List successful runs of the workflow whose file path ends with `workflow_filename`,
    created at or after `since_dt` (a timezone-aware datetime).
    """
    headers = {'Accept': 'application/vnd.github+json'}
    if token:
        headers['Authorization'] = f"Bearer {token}"

    # 1) Fetch all workflows and find the matching one by path
    url_list = f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows"
    resp = request_with_retries(url_list, headers=headers)
    data = resp.json().get('workflows', [])
    wf_id = None
    for wf in data:
        # wf['path'] is like '.github/workflows/run-runner.yml'
        if wf.get('path', '').endswith("/" + workflow_filename):
            wf_id = wf['id']
            break
    if wf_id is None:
        raise RuntimeError(f"Workflow '{workflow_filename}' not found in repo {owner}/{repo}")

    # 2) Paginate through runs for that workflow ID
    runs = []
    page = 1
    while True:
        url_runs = (
            f"{GITHUB_API}/repos/{owner}/{repo}"
            f"/actions/workflows/{wf_id}/runs"
        )
        params = {'status':'success','per_page':100,'page':page}
        resp = request_with_retries(url_runs, headers=headers, params=params)
        batch = resp.json().get('workflow_runs', [])
        if not batch:
            break

        # collect only those newer than since_dt
        for run in batch:
            created = datetime.fromisoformat(
                run['created_at'].replace('Z', '+00:00')
            )
            if created >= since_dt:
                runs.append(run)
            else:
                # runs are descending by date, so we can stop paging
                return runs
        page += 1

    return runs


def get_artifacts(owner, repo, run_id, token):
    headers = {'Accept': 'application/vnd.github+json'}
    if token: headers['Authorization'] = f"Bearer {token}"
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    return request_with_retries(url, headers=headers).json().get('artifacts', [])

def download_and_extract(owner, repo, artifact, token, base_dir):
    """Download one artifact, unzip into base_dir/run_<id>/artifact_<id>/"""
    headers = {'Accept': 'application/vnd.github+json'}
    if token: headers['Authorization'] = f"Bearer {token}"

    run_id = artifact['workflow_run']['id']
    art_id = artifact['id']
    name   = artifact['name']
    dl_url = artifact['archive_download_url']

    run_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    zip_path = os.path.join(run_dir, f"{name}_{art_id}.zip")
    with request_with_retries(dl_url, headers=headers, stream=True).raw as r:
        with open(zip_path, 'wb') as f:
            for chunk in tqdm(r, desc=f"Downloading {name}", unit='KB', unit_scale=True):
                f.write(chunk)

    # extract
    ext_dir = os.path.join(run_dir, f"artifact_{art_id}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(ext_dir)
    os.remove(zip_path)
    return ext_dir

def collect_csv_paths(extracted_dir):
    """
    Look under extracted_dir for any CSVs in a path matching:
       .../results/L{Lx}_Ly{Ly}_T{time}/correlator_*.csv
    Return list of tuples: (csv_path, Lx, time).
    """
    pattern = re.compile(r"L(\d+)_Ly\d+_T(\d+)")
    found = []
    for root, _, files in os.walk(extracted_dir):
        for fn in files:
            if fn.endswith(".csv"):
                m = pattern.search(root)
                if not m:
                    continue
                Lx, T = m.group(1), m.group(2)
                found.append((os.path.join(root, fn), Lx, T))
    return found

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--owner',    default="alessum")
    p.add_argument('--repo',     default="quera_proj")
    p.add_argument('--workflow', default="run-runner.yml")
    p.add_argument('--hours',    type=float, default=12)
    p.add_argument('--outdir',   default="results")
    p.add_argument('--token',    default=None)
    args = p.parse_args()

    token = "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"
    if not token:
        print("ERROR: set GITHUB_TOKEN or pass --token", file=sys.stderr)
        sys.exit(1)

    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    runs = get_workflow_runs(args.owner, args.repo, args.workflow, token, since)
    if not runs:
        print("No runs found.")
        return

    tmp_base = "tmp_fetch"
    if os.path.exists(tmp_base):
        shutil.rmtree(tmp_base)
    os.makedirs(tmp_base)

    # 1) Download & extract all artifacts
    for run in runs:
        arts = get_artifacts(args.owner, args.repo, run['id'], token)
        for art in arts:
            download_and_extract(args.owner, args.repo, art, token, tmp_base)

    # 2) Walk extracted dirs, gather CSVs with metadata
    csvs = []
    for entry in os.scandir(tmp_base):
        if entry.is_dir():
            csvs += collect_csv_paths(entry.path)

    # 3) Group by (Lx, T) and concat into results/L{Lx}/T{T}/combined.csv
    for (Lx, T), group in tqdm(
        groupby(sorted(csvs, key=lambda x:(x[1], x[2])),
                key=lambda x:(x[1], x[2])),
        desc="Writing output"
    ):
        out_dir = os.path.join(args.outdir, f"L{Lx}", f"T{T}")
        os.makedirs(out_dir, exist_ok=True)
        dfs = [pd.read_csv(path, header=None) for path,_,_ in group]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(os.path.join(out_dir, "combined.csv"), index=False, header=False)

    print(f"Done. Final organized results in '{args.outdir}/'")

if __name__=="__main__":
    main()
