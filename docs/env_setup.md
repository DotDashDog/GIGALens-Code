# Environment setup

This repository is run on **NERSC Perlmutter** inside a pinned NVIDIA Shifter
container that ships JAX 0.10, layered on top of a conda env that supplies
everything else (lenstronomy, blackjax, matplotlib, etc.), with a small
sidecar directory of overlay packages (TFP-nightly and a newer astropy)
that the container needs but the conda env doesn't ship.

The exact layout below is what notebooks expect; if you change any piece,
JAX / TFP / NumPy versions can silently drift and break the pipeline.

## Canonical runtime

| Layer | Value |
|---|---|
| Shifter image | `docker:ghcr.io/nvidia/jax:jax-2026-04-13` |
| Conda env | `gigalens_multinode_env` (Python 3.12) |
| Sidecar overlay | `/global/homes/l/linusu/sidecar_jax_upgrade` (TFP-nightly, astropy>=7.2) |
| Jupyter kernel | `GigalensMultiNode (JAX 2026)` (id `gigalens_multinode_env_newjax`) |

Expected versions inside the container:

- `jax >= 0.10.0.dev20260505`
- `tensorflow_probability >= 0.26.0-dev20260505`
- `numpy >= 2.4`
- `astropy >= 7.2`

## PYTHONPATH ordering

The sidecar **must come first** so that its `tfp-nightly` shadows the
older `tensorflow_probability 0.25` shipped by the conda env. The
canonical export line (already baked into the kernel.json) is:

```bash
export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
```

If you swap that order, you get NumPy 2.x compatibility errors and JAX 0.7+
will refuse to import TFP.

## Making the two source repos importable (no sys.path hacks)

Both `gigalens` (the upstream library, kept as a sibling git checkout at
`~/gigalens/`) and `gigalens_research` (this repo) should be installed
**editable** into the conda env exactly once. After that, `import gigalens`
and `import gigalens_research` work from any notebook or script without
touching `sys.path`.

Run these two commands **inside the Shifter container**, since that's where
the kernel's Python lives:

```bash
shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 bash
# Now inside the container:
export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
PIP="$HOME/.conda/envs/gigalens_multinode_env/bin/pip"

$PIP install --no-deps -e ~/gigalens
$PIP install --no-deps -e ~/GIGALens-Code
```

`--no-deps` is critical: it stops pip from trying to "helpfully" upgrade
JAX, TFP, NumPy, etc. The conda env + sidecar already provides everything
both packages need. The editable install is just a `.pth` file pointer.

Verify (still inside the container, or via the Jupyter kernel):

```bash
python -c "import gigalens, gigalens_research; print(gigalens.__file__, gigalens_research.__file__)"
```

Both paths should point under your home directory (not site-packages).

### Which `python` runs code/tests (important)

Inside the container, run code and pytest with the **container interpreter**
`/usr/bin/python3` — that is the one that ships JAX 0.10. The conda env's own
`python` (and the `pytest` on its `PATH`) has **no jax**; the conda env only
supplies the non-JAX deps (lenstronomy, tfp, etc.) via `PYTHONPATH`. So a bare
`python -m pytest` can silently pick the wrong interpreter depending on `PATH`.
Be explicit:

```bash
# inside the container, with PYTHONPATH set as above (+ ~/gigalens/src etc.):
/usr/bin/python3 -m pytest gigalens/tests/validation -q
```

`pip` (for the editable install above) is the conda env's pip on purpose — it
writes the `.pth`; only the *run* interpreter must be the container's python.

## Why editable install instead of sys.path

Three reasons:

1. `from gigalens_research.inference import run_pipeline` works from any
   directory — no `sys.path.insert(0, ...)` boilerplate at the top of every
   notebook / script.
2. Renames and refactors inside the package don't require updating any
   notebook — the import path tracks the file layout.
3. Scripts launched by slurm or `python -m` don't need extra environment
   setup; the `.pth` file is read automatically at interpreter startup.

## Legacy runtime (JAX 0.6 / 2025)

Only use this when reproducing pre-upgrade behavior:

- Kernel: `GigalensMultiNode`
- Image: `docker:ghcr.io/nvidia/jax:jax-2025-06-07`
- PYTHONPATH: only the conda env, no sidecar
- Known issue: XLA GEMM-autotune `LOG(FATAL)` crashes `MCLMC_JIT` on
  DESI-238.
