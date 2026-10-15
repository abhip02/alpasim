# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Compare regular and sharded rollout outputs without loading pickled debug data."""

import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path

from alpasim_grpc.v0.logging_pb2 import LogEntry


def read_run(directory: Path) -> dict:
    paths = list((directory / "rollouts").rglob("rollout.asl"))
    if len(paths) != 1:
        raise ValueError(f"Expected one rollout.asl in {directory}, found {len(paths)}")
    records = collections.defaultdict(list)
    with paths[0].open("rb") as stream:
        while header := stream.read(4):
            (size,) = struct.unpack(">L", header)
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError(f"Truncated ASL record in {paths[0]}")
            entry = LogEntry.FromString(payload)
            kind = entry.WhichOneof("log_entry")
            records[kind].append(getattr(entry, kind))
    return records


def compare(regular: Path, sharded: Path) -> dict:
    a, b = read_run(regular), read_run(sharded)
    if (
        not a["driver_return"]
        or not a["actor_poses"]
        or not a["driver_session_request"]
    ):
        raise ValueError(
            "The regular rollout has no policy outputs, ego poses, or session"
        )
    inputs_equal = {}
    for kind in (
        "driver_camera_image",
        "driver_ego_trajectory",
        "route_request",
        "driver_request",
    ):
        hashes = []
        for run in (a, b):
            values = []
            for value in run[kind]:
                copied = type(value)()
                copied.CopyFrom(value)
                copied.ClearField("session_uuid")
                values.append(
                    hashlib.sha256(
                        copied.SerializeToString(deterministic=True)
                    ).hexdigest()
                )
            hashes.append(sorted(values))
        inputs_equal[kind] = hashes[0] == hashes[1]
    trajectories_equal = len(a["driver_return"]) == len(b["driver_return"]) and all(
        x.trajectory == y.trajectory
        for x, y in zip(a["driver_return"], b["driver_return"])
    )

    def ego(run):
        return [
            (
                v.timestamp_us,
                next(p.actor_pose for p in v.actor_poses if p.actor_id == "EGO"),
            )
            for v in run["actor_poses"]
        ]

    metrics = [
        json.loads((directory / "aggregate/results-summary.json").read_text())[
            "metrics_results"
        ][0]
        for directory in (regular, sharded)
    ]
    metrics = [
        {k: v for k, v in row.items() if k not in ("run_name", "run_uuid")}
        for row in metrics
    ]
    seeds = [[v.random_seed for v in run["driver_session_request"]] for run in (a, b)]
    result = {
        "session_seeds": seeds,
        "session_seeds_equal": seeds[0] == seeds[1],
        "input_records_equal_ignoring_session_ids_and_order": inputs_equal,
        "policy_call_counts": [len(run["driver_return"]) for run in (a, b)],
        "trajectories_exactly_equal": trajectories_equal,
        "ego_pose_counts": [len(run["actor_poses"]) for run in (a, b)],
        "ego_poses_exactly_equal": ego(a) == ego(b),
        "aggregate_metrics_equal_excluding_run_ids": metrics[0] == metrics[1],
    }
    result["pass"] = all(
        (
            result["session_seeds_equal"],
            all(inputs_equal.values()),
            trajectories_equal,
            result["ego_poses_exactly_equal"],
            result["aggregate_metrics_equal_excluding_run_ids"],
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("regular", type=Path)
    parser.add_argument("sharded", type=Path)
    args = parser.parse_args()
    result = compare(args.regular, args.sharded)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
