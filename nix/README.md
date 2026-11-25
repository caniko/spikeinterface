# SpikeInterface with Nix

The flake targets `x86_64-linux` with Python 3.13. Its packages are immutable
virtual environments; run their `bin/python` directly.

| Package | Dev shell | PyTorch |
| --- | --- | --- |
| `spikeinterface` (`default`) | — | Not installed |
| `lupin-cpu` | `cpu` (`default`) | CPU |
| `lupin-amd` (`lupin-rocm`) | `amd` (`rocm`) | ROCm 7.2 |
| `lupin-nvidia` (`lupin-cuda`) | `nvidia` (`cuda`) | CUDA 12.8 |

```sh
nix build .#lupin-amd
./result/bin/python nix/smoke.py cuda --motion
nix develop .#amd -c python nix/smoke.py cuda
```

Dev shells put the corresponding immutable environment on `PATH`; they do
not create a `.venv` or run `uv sync`. AMD uses the wheel's bundled ROCm
libraries and needs access to `/dev/kfd` and the GPU's render device. NVIDIA
needs a compatible host driver under `/run/opengl-driver/lib` on NixOS.
`nix flake check` runs the CPU smoke test; `nix build .#lupin-nvidia-check`
checks CUDA imports and ELF linkage without a GPU.
The NVIDIA environment does not ship NVSHMEM's optional distributed transport
dependencies (Open MPI/OpenSHMEM, PMIx, libfabric, and UCX); Lupin does not use them.

To run Lupin with GPU template matching on either vendor:

```python
from spikeinterface.sorters import run_sorter

sorting = run_sorter(
    "lupin",
    recording,
    folder="lupin-output",
    job_kwargs={"n_jobs": 1},
    template_matching_kwargs={"engine": "torch", "torch_device": "cuda"},
)
```

Other sorter stages are not moved to the GPU by this option. Add
`apply_motion_correction=True` to enable DREDGE.

## Downstream environments

With this flake available as `spikeinterface`, build an environment with
Lupin and HDF5 NWB support (including `stream_mode="remfile"` for remote NWB
URLs such as DANDI asset URLs):

```nix
spikeinterface.lib.x86_64-linux.mkEnv {
  name = "my-lupin-env";
  extras = ["lupin" "nwb" "amd"];
}
```

Choose exactly one of `cpu`, `amd`, or `nvidia` with `lupin`. Pass `pkgs`
optionally to use your own `x86_64-linux` package set. To refresh Python
dependencies, run `uv lock --project nix` with Python 3.13 available; the
upstream root `pyproject.toml` is not changed by the Nix setup.
