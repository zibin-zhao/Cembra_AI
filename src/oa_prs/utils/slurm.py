"""
SLURM job submission and monitoring utilities.
"""

import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from oa_prs.utils.logging_config import get_logger

log = get_logger(__name__)


def generate_slurm_header(config: dict[str, Any]) -> str:
    """
    Generate SLURM batch script header from configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary. Expected keys:
        - job_name: str
        - time: str (HH:MM:SS or minutes)
        - nodes: int (default 1)
        - cpus_per_task: int
        - mem_per_node: str (e.g., "32G")
        - partition: str (e.g., "gpu", "cpu")
        - gpus: Optional[int]
        - email: Optional[str]
        - email_type: str (NONE, BEGIN, END, FAIL, ALL)
        - output: Optional[str] (log file path)
        - error: Optional[str] (error log path)

    Returns
    -------
    str
        SLURM header string with #SBATCH directives

    Examples
    --------
    >>> config = {
    ...     "job_name": "my_job",
    ...     "time": "01:00:00",
    ...     "cpus_per_task": 8,
    ...     "mem_per_node": "16G",
    ...     "partition": "gpu",
    ...     "gpus": 1,
    ... }
    >>> header = generate_slurm_header(config)
    """
    lines = ["#!/bin/bash"]
    lines.append(f"#SBATCH --job-name={config.get('job_name', 'job')}")

    # Time
    if "time" in config:
        time_str = config["time"]
        if not re.match(r"^\d+:\d{2}:\d{2}$", time_str):
            # Assume minutes, convert to HH:MM:SS
            minutes = int(time_str)
            hours = minutes // 60
            mins = minutes % 60
            time_str = f"{hours:02d}:{mins:02d}:00"
        lines.append(f"#SBATCH --time={time_str}")

    # Resources
    lines.append(f"#SBATCH --nodes={config.get('nodes', 1)}")
    if "cpus_per_task" in config:
        lines.append(f"#SBATCH --cpus-per-task={config['cpus_per_task']}")
    if "mem_per_node" in config:
        lines.append(f"#SBATCH --mem={config['mem_per_node']}")

    # Partition and GPU
    if "partition" in config:
        lines.append(f"#SBATCH --partition={config['partition']}")
    if "gpus" in config:
        lines.append(f"#SBATCH --gpus={config['gpus']}")

    # Notifications
    if "email" in config:
        lines.append(f"#SBATCH --mail-user={config['email']}")
        lines.append(
            f"#SBATCH --mail-type={config.get('email_type', 'NONE')}"
        )

    # Output files
    if "output" in config:
        lines.append(f"#SBATCH --output={config['output']}")
    if "error" in config:
        lines.append(f"#SBATCH --error={config['error']}")

    return "\n".join(lines) + "\n"


def submit_job(script_path: str | Path) -> str:
    """
    Submit a SLURM job script and return job ID.

    Parameters
    ----------
    script_path : str | Path
        Path to SLURM batch script

    Returns
    -------
    str
        SLURM job ID (e.g., "12345")

    Raises
    ------
    FileNotFoundError
        If script does not exist
    RuntimeError
        If submission fails

    Examples
    --------
    >>> job_id = submit_job("script.sh")
    >>> print(f"Submitted job {job_id}")
    """
    script_path = Path(script_path)

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse job ID from output (e.g., "Submitted batch job 12345")
        match = re.search(r"job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            log.info("submitted_job", job_id=job_id, script=str(script_path))
            return job_id
        else:
            raise RuntimeError(f"Could not parse job ID from: {result.stdout}")
    except subprocess.CalledProcessError as e:
        log.error(
            "job_submission_failed",
            script=str(script_path),
            stderr=e.stderr,
        )
        raise RuntimeError(f"sbatch submission failed: {e.stderr}") from e


def check_job_status(job_id: str) -> str:
    """
    Check SLURM job status.

    Parameters
    ----------
    job_id : str
        SLURM job ID

    Returns
    -------
    str
        Job status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, etc.)

    Raises
    ------
    RuntimeError
        If job not found or squeue fails

    Examples
    --------
    >>> status = check_job_status("12345")
    >>> print(f"Job status: {status}")
    """
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-o", "%T"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            # Job not found in queue (might be completed)
            # Try sacct for historical info
            result = subprocess.run(
                ["sacct", "-j", job_id, "-o", "State", "-n"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                status = result.stdout.strip().split()[0]
                return status
            else:
                raise RuntimeError(f"Job {job_id} not found")

        # squeue output includes header, get second line
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            return lines[1].strip()
        else:
            raise RuntimeError(f"Could not parse status for job {job_id}")
    except Exception as e:
        log.error("job_status_check_failed", job_id=job_id, error=str(e))
        raise


def cancel_job(job_id: str) -> bool:
    """
    Cancel a SLURM job.

    Parameters
    ----------
    job_id : str
        SLURM job ID

    Returns
    -------
    bool
        True if cancellation succeeded

    Examples
    --------
    >>> cancelled = cancel_job("12345")
    """
    try:
        subprocess.run(
            ["scancel", job_id],
            check=True,
            capture_output=True,
        )
        log.info("job_cancelled", job_id=job_id)
        return True
    except subprocess.CalledProcessError as e:
        log.error("job_cancellation_failed", job_id=job_id, error=e.stderr)
        return False
