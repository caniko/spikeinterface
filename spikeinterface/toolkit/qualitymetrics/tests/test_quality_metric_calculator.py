import unittest
import shutil
from pathlib import Path

import pytest

from spikeinterface import WaveformExtractor
from spikeinterface.extractors import toy_example

from spikeinterface.toolkit.postprocessing import WaveformPrincipalComponent
from spikeinterface.toolkit.qualitymetrics import compute_quality_metrics


def setup_module():
    for folder in ('toy_rec', 'toy_sorting', 'toy_waveforms'):
        if Path(folder).is_dir():
            shutil.rmtree(folder)

    recording, sorting = toy_example(num_segments=2, num_units=10)
    recording = recording.save(folder='toy_rec')
    sorting = sorting.save(folder='toy_sorting')

    we = WaveformExtractor.create(recording, sorting, 'toy_waveforms')
    we.set_params(ms_before=3., ms_after=4., max_spikes_per_unit=500)
    we.run(n_jobs=1, chunk_size=30000)


def test_compute_quality_metrics():
    we = WaveformExtractor.load_from_folder('toy_waveforms')
    print(we)

    # without PC
    metrics = compute_quality_metrics(we, metric_names=['snr'])
    print(metrics)
    print(metrics.columns)

    # with PCs
    pca = WaveformPrincipalComponent(we)
    pca.set_params(n_components=5, mode='by_channel_local')
    pca.run()
    metrics = compute_quality_metrics(we, waveform_principal_component=pca)
    print(metrics)
    print(metrics.columns)


if __name__ == '__main__':
    setup_module()
    test_compute_quality_metrics()
