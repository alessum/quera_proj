import subprocess
import time
import numpy as np

def run_workflows(
    thetas=[0.3, .35, .4, .45, .5],
    circuits=np.arange(0, 10, 3),
    ref="main",
    number_of_circuits=3,
    time_to_run=10000,
    workflow_file="run-runner.yml",
    max_retries=3,
    retry_delay=5  # seconds to wait before retry
):
    for circuit in circuits:
        for theta in thetas:
            print(f"Running workflow for circuit={circuit}, theta={theta}…")
            for attempt in range(1, max_retries + 1):
                try:
                    subprocess.run(
                        [
                            "gh", "workflow", "run", workflow_file,
                            "--ref", str(ref),
                            "-f", f"circuit_to_run={circuit}",
                            "-f", f"number_of_circuits={number_of_circuits}",
                            "-f", f"theta_to_run={theta}",
                            "-f", f"time_to_run={time_to_run}"
                        ],
                        check=True,
                        timeout=5  # optionally fail if no response in 15s
                    )
                    # if we get here, it succeeded
                    print("  → succeeded")
                    break
                except subprocess.TimeoutExpired:
                    print(f"  ⚠️ attempt {attempt} timed out; retrying in {retry_delay}s…")
                except subprocess.CalledProcessError as e:
                    print(f"  ⚠️ attempt {attempt} failed (exit {e.returncode}); retrying in {retry_delay}s…")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    # all attempts exhausted
                    raise RuntimeError(
                        f"Workflow for circuit={circuit}, theta={theta} failed after {max_retries} attempts"
                    )

if __name__ == "__main__":
    run_workflows()
