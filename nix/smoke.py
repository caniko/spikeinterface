"""Exercise the packaged Lupin sorter without downloading test data."""

import argparse
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import spikeinterface as si
import torch
from spikeinterface.sorters import LupinSorter, run_sorter


def smoke_sort(device, motion):
    from spikeinterface.sortingcomponents.matching import wobble

    torch.set_num_threads(1)
    if device == "cuda":
        assert torch.cuda.is_available()
        assert torch.version.hip or torch.version.cuda
        assert torch.arange(4, device=device).square().tolist() == [0, 1, 4, 9]
        print(f"{torch.cuda.get_device_properties(0)}; HIP={torch.version.hip}; CUDA={torch.version.cuda}")
    else:
        assert torch.version.cuda is None and torch.version.hip is None

    conv1d = wobble.conv1d
    calls = 0

    def checked_conv1d(traces, weights, *args, **kwargs):
        nonlocal calls
        assert traces.device.type == weights.device.type == device
        result = conv1d(traces, weights, *args, **kwargs)
        assert result.device.type == device
        calls += 1
        return result

    with TemporaryDirectory(prefix="lupin-smoke-") as directory:
        folder = Path(directory)
        rec, _ = si.generate_ground_truth_recording(
            durations=[20.0 if motion else 10.0], num_channels=16 if motion else 4, seed=0
        )
        rec = rec.save(folder=folder / "recording", format="binary", n_jobs=1, progress_bar=False)
        with patch.object(wobble, "conv1d", checked_conv1d):
            sorting = run_sorter(
                "lupin",
                rec,
                folder=folder / "lupin",
                seed=0,
                job_kwargs={"n_jobs": 1},
                template_matching_kwargs={"engine": "torch", "torch_device": device},
                apply_motion_correction=motion,
            )
        units = sorting.get_num_units()
        spikes = sorting.to_spike_vector().size
        assert units > 0 and spikes > 0 and calls > 0
        output = folder / "lupin" / "sorter_output"
        analyzer = si.load_sorting_analyzer(output / "analyzer")
        assert analyzer.sorting.get_num_units() == units
        if motion:
            assert (output / "motion").is_dir()
        print(f"Lupin: {units} units, {spikes} spikes, {calls} Wobble convolutions on {device}; analyzer reloaded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "cuda", "nvidia-offline"))
    parser.add_argument("--motion", action="store_true", help="enable Lupin's DREDGE motion correction")
    args = parser.parse_args()
    assert LupinSorter.is_installed()
    print(f"SpikeInterface {si.__version__}; PyTorch {torch.__version__}; LupinSorter.is_installed() = True")
    if args.mode == "nvidia-offline":
        if args.motion:
            parser.error("--motion requires cpu or cuda")
        for module in ("scipy", "numba", "sklearn"):
            print(f"{module} {importlib.import_module(module).__version__}")
        assert torch.version.cuda and torch.version.hip is None
        cudnn = torch.backends.cudnn.version()
        assert isinstance(cudnn, int) and cudnn > 0
        print(f"cuDNN {cudnn}")
        assert not torch.cuda.is_available()
        print(f"CUDA {torch.version.cuda}; torch.cuda.is_available() = False")
    else:
        smoke_sort(args.mode, args.motion)


if __name__ == "__main__":
    main()
