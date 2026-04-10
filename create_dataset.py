
from concurrent.futures import ProcessPoolExecutor
import os
import shutil

from torchsig.utils.printing import dataset_metadata_str
from torchsig.utils.defaults import TorchSigDefaults, SR
from torchsig.transforms.impairments import Impairments
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import DatasetCreator

ROOT = "/scratch/sdrdata/hdf5"
DATASET_LENGTH = 1000
MAX_WORKERS = 8

shutil.rmtree(ROOT)
os.mkdir(ROOT)

def create_dataset(max_signals, impairments_level):
    dataset_metadata = TorchSigDefaults().default_dataset_metadata
    dataset_metadata["num_signals_min"] = max_signals
    dataset_metadata["num_signals_max"] = max_signals
    dataset_metadata["bandwidth_min"] = 1e4
    dataset_metadata["bandwidth_max"] = 5e5
    dataset_metadata["cochannel_overlap_probability"] = 0
    impairments = Impairments(level=impairments_level)
    burst_impairments = impairments.signal_transforms
    whole_signal_impairments = impairments.dataset_transforms

    dataset = TorchSigIterableDataset(
        metadata=dataset_metadata,
        transforms=[
            whole_signal_impairments,
        ],
        component_transforms=[burst_impairments],
    )

    dataloader = WorkerSeedingDataLoader(dataset, batch_size=2)

    dataset_creator = DatasetCreator(
        dataset_length=DATASET_LENGTH,
        dataloader=dataloader,
        root=f"/scratch/sdrdata/hdf5/nb-maxsig-{max_signals}-imp{impairments_level}",
        overwrite=True,
        multithreading=False,
    )
    dataset_creator.create()


with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = []
    for max_signals in (1, 4):
        futures.append(pool.submit(create_dataset, max_signals, impairments_level=2))
    for future in futures:
        if future.exception():
            print(future.exception())
