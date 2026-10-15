# Alpamayo 1.5 on two GPUs

`driver=alpamayo1_5_sharded` splits one Alpamayo 1.5 model across two GPUs
inside one driver container. The regular `alpamayo1_5` preset keeps single-GPU
loading. This is experimental sequential model parallelism: it reduces resident
weights per GPU, without introducing concurrent pipeline or tensor parallelism.

The driver packages the tested device map alongside the adapter. VLM language
layers 0–19 and their matching action-expert layers run on local GPU 0; layers
20–35 run on local GPU 1. Matching layer placement matters because the action
expert reuses the VLM's KV cache. Plain `device_map="auto"` failed with a
cross-device cache update in the original experiment.

Weights remain BF16, and both attention paths use SDPA. Inference, sampling,
trajectory conversion, and controller behavior use their existing code. The
loader checks a 12 GiB resident parameter-and-buffer budget on each GPU. This
is not a total VRAM cap: activations, caches, CUDA libraries, and allocator
reservations need additional space. The tested cards were two 24 GB RTX A5000s.
The map targets the 36-layer `nvidia/Alpamayo-1.5-10B` architecture; revalidate
placement and numerical behavior after model or dependency changes.

## Launch locally

Follow [ONBOARDING.md](ONBOARDING.md) for image, model-cache, and scene setup.
Install the driver dependencies, including Accelerate, when building the service
image. No experiment-directory mounts or replacement adapter files are needed.

The wizard currently creates one driver container per entry in
`services.driver.gpus`. Generate one driver, then expose both model GPUs to that
same container in the generated Compose file. Use a fresh run directory and
your usual image and scene overrides. For a machine using host GPUs 0 and 1
for the model and GPU 2 for rendering/physics:

```bash
export ALPASIM_RUN_DIR="$PWD/outputs/sharded-example"
uv run alpasim_wizard deploy=local topology=1gpu driver=alpamayo1_5_sharded \
  wizard.run_method=NONE "wizard.log_dir=$ALPASIM_RUN_DIR" \
  services.driver.gpus=[0] services.driver.replicas_per_container=1 \
  services.renderer.gpus=[2] services.physics.gpus=[2] \
  services.trafficsim.gpus=null \
  runtime.nr_workers=1 runtime.endpoints.driver.n_concurrent_rollouts=1
```

Find the two selected model GPU UUIDs with
`nvidia-smi --query-gpu=index,uuid --format=csv,noheader`, and set their order:

```bash
export ALPASIM_DRIVER_GPU_UUIDS=GPU-REPLACE-FIRST,GPU-REPLACE-SECOND
uv run python - <<'PY'
import os
from pathlib import Path
import yaml

uuids = os.environ["ALPASIM_DRIVER_GPU_UUIDS"].split(",")
assert len(uuids) == 2 and len(set(uuids)) == 2
assert all(u.startswith("GPU-") and "REPLACE" not in u for u in uuids)
path = Path(os.environ["ALPASIM_RUN_DIR"]) / "docker-compose.yaml"
compose = yaml.safe_load(path.read_text())
assert [k for k in compose["services"] if k.startswith("driver-")] == ["driver-0"]
driver = compose["services"]["driver-0"]
driver["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] = uuids
env = driver.setdefault("environment", {})
if isinstance(env, list):
    env = dict(item.split("=", 1) for item in env)
env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
driver["environment"] = env
path.write_text(yaml.safe_dump(compose, sort_keys=False))
PY

docker compose -f "$ALPASIM_RUN_DIR/docker-compose.yaml" config --quiet
docker compose -f "$ALPASIM_RUN_DIR/docker-compose.yaml" up --abort-on-container-exit --exit-code-from runtime-0
docker compose -f "$ALPASIM_RUN_DIR/docker-compose.yaml" down
```

UUID ordering makes local GPU 0 explicit. The input device must be `cuda:0`,
and exactly two CUDA GPUs must be visible to the driver. The preset also uses
CPU image decoding, matching the controlled comparison. Choose GPU allocations
and service concurrency appropriate to available memory.

## Recorded numerical validation

On October 5, 2026, regular and sharded runs used the same scene
`clipgt-01d503d4-449b-46fc-8d78-9085e70d3554` (NuRec 26.04), session seed 42,
`force_determinism=true`, BF16, SDPA, CPU image decoding, batch size one, and one
trajectory sample. Both completed 25 simulation steps with eight policy calls.

| Compared output | Result |
| --- | --- |
| Eight predicted trajectories, 65 poses each | Exact protobuf equality; 0 m position difference |
| Eight reasoning texts | Exact equality |
| 27 recorded ego poses | Exact equality; 0 m position difference |
| Aggregate metric values | Exact equality excluding run identifiers |

Rechecking the raw logs also confirmed all 104 camera records, 25 ego
observations, 25 route requests, and eight driver requests match after ignoring
session UUIDs and asynchronous log ordering. This is evidence for one short
rollout, including 1.7 seconds of ground-truth warmup and about 0.8 seconds
following warmup. It does not prove equality across all scenes, seeds, longer
runs, attention backends, or hardware. The integrated loader preserves the
archived map and loading arguments; these recorded GPU results come from the
original experimental adapter, not a new rollout of this branch.

Tested versions: AlpaSim `affc2eab209fa43bdfa2f26c0f8d437922d78a68`,
Alpamayo 1.5 `36aeb4c5938cbc2eb2aed33b22434773da4ab639`, model snapshot
`7aba8293c09993f2e125c6819df05d7fa3e873ea`, PyTorch 2.8.0,
Transformers 4.57.1, and Accelerate 1.15.0 in the driver container.
The loader does not pin the checkpoint revision: set `driver.model.checkpoint_path`
to a cached snapshot path inside the container for exact weight reproduction.

The compact original report is [comparison-seed42.json](../tools/sharding/comparison-seed42.json).
Raw logs and videos remain outside Git in the original local archive:
`/scratch/apomalap/thesis/experiments/alpasim-sharding/comparison-seed42/`.

Recheck any saved pair without simulation or GPU access:

```bash
uv run python tools/sharding/compare_rollouts.py /path/to/regular /path/to/sharded
```

The command prints JSON and exits nonzero on mismatches. It compares session
seeds, input records, full trajectory protobufs, ego poses, and aggregate metrics.
It does not deserialize the pickled debug records containing reasoning text;
reasoning equality above is retained from the original trusted-local-log check.
