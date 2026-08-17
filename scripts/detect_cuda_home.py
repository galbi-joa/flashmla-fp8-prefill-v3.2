#!/usr/bin/env python3
"""Select a CUDA Toolkit installation suitable for the SM100 build."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


MINIMUM_VERSION = (12, 9)


def candidate_homes() -> list[Path]:
    candidates: list[Path] = []

    if cuda_home := os.environ.get("CUDA_HOME"):
        candidates.append(Path(cuda_home))

    if nvcc := shutil.which("nvcc"):
        candidates.append(Path(nvcc).resolve().parent.parent)

    candidates.append(Path("/usr/local/cuda"))
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def nvcc_version(cuda_home: Path) -> tuple[int, int] | None:
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        return None
    try:
        output = subprocess.check_output(
            [str(nvcc), "--version"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def pytorch_cuda_version() -> tuple[int, int] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.version.cuda:
        return None
    major, minor, *_ = torch.version.cuda.split(".")
    return int(major), int(minor)


def select_cuda_home() -> tuple[Path, tuple[int, int]]:
    discovered = [
        (home, version)
        for home in candidate_homes()
        if (version := nvcc_version(home)) is not None
    ]
    supported = [item for item in discovered if item[1] >= MINIMUM_VERSION]
    preferred = pytorch_cuda_version()

    if not supported:
        if discovered:
            details = ", ".join(
                f"CUDA {major}.{minor} at {home}"
                for home, (major, minor) in discovered
            )
            print(f"Unsupported CUDA Toolkit installation(s): {details}", file=sys.stderr)
        else:
            print("No CUDA Toolkit installation containing nvcc was found.", file=sys.stderr)
        print(
            "SM100 builds require nvcc 12.9 or newer. "
            "Install cuda-toolkit-12-9 or cuda-toolkit-13-0.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if preferred is None and (explicit := os.environ.get("CUDA_HOME")):
        explicit_home = Path(explicit).resolve()
        for item in supported:
            if item[0] == explicit_home:
                return item

    if preferred is not None:
        exact = [item for item in supported if item[1] == preferred]
        if exact:
            return exact[0]

        same_major = [item for item in supported if item[1][0] == preferred[0]]
        if same_major:
            return min(same_major, key=lambda item: abs(item[1][1] - preferred[1]))

    return max(supported, key=lambda item: item[1])


def main() -> None:
    cuda_home, (major, minor) = select_cuda_home()
    print(f"Selected CUDA {major}.{minor} at {cuda_home}", file=sys.stderr)
    print(cuda_home)


if __name__ == "__main__":
    main()
