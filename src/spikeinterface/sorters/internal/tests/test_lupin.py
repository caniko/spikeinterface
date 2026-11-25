import unittest
from unittest.mock import patch


from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite

from spikeinterface.sorters import LupinSorter, run_sorter
from spikeinterface.sortingcomponents.matching import wobble

from pathlib import Path


class LupinSorterCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = LupinSorter

    # TODO: remove once writing to zarr is supported with zarr>=3 (save_array writes the templates to zarr)

    def test_with_run(self):
        super().test_with_run()

    def test_template_matching_kwargs_none(self):
        sorting = run_sorter(
            "lupin",
            self.recording,
            folder=self.cache_folder / "lupin-default-matching",
            job_kwargs={"n_jobs": 1},
            seed=0,
            template_matching_kwargs=None,
        )
        assert sorting.get_num_units() > 0

    def test_template_matching_kwargs(self):
        with patch.object(wobble, "compute_objective", wraps=wobble.compute_objective) as objective:
            sorting = run_sorter(
                "lupin",
                self.recording,
                folder=self.cache_folder / "lupin-torch",
                job_kwargs={"n_jobs": 1},
                seed=0,
                template_matching_kwargs={"engine": "torch", "torch_device": "cpu"},
            )

        assert sorting.get_num_units() > 0
        assert sorting.to_spike_vector().size > 0
        assert objective.call_count > 0
        assert all(call.args[-2:] == ("torch", "cpu") for call in objective.call_args_list)


if __name__ == "__main__":
    test = LupinSorterCommonTestSuite()
    test.cache_folder = Path(__file__).resolve().parents[4] / "cache_folder" / "sorters"
    test.setUp()
    test.test_with_run()
