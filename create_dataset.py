
from concurrent.futures import ProcessPoolExecutor
import os
import shutil

from torchsig.utils.printing import dataset_metadata_str
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.transforms.impairments import Impairments
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import DatasetCreator

ROOT = "/scratch/sdrdata/hdf5"
DATASET_LENGTH = 10
MAX_WORKERS = 8

shutil.rmtree(ROOT)
os.mkdir(ROOT)

def create_dataset(max_signals, impairments_level, signal_generators):
    dataset_metadata = TorchSigDefaults().default_dataset_metadata
    dataset_metadata["max_signals"] = max_signals
    impairments = Impairments(level=impairments_level)
    burst_impairments = impairments.signal_transforms
    whole_signal_impairments = impairments.dataset_transforms

    dataset = TorchSigIterableDataset(
        signal_generators = signal_generators,
        metadata=dataset_metadata,
        transforms=[
            whole_signal_impairments,
        ],
        component_transforms=[burst_impairments],
    )
    dataloader = WorkerSeedingDataLoader(dataset, batch_size=2)
    signal_generator_names = "-".join(signal_generators)

    dataset_creator = DatasetCreator(
        dataset_length=DATASET_LENGTH,
        dataloader=dataloader,
        root=f"/scratch/sdrdata/hdf5/{signal_generator_names}-maxsig-{max_signals}-imp{impairments_level}",
        overwrite=True,
        multithreading=False,
    )
    dataset_creator.create()


create_dataset(1, 0, ["tone"])
raise

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = []
    for max_signals in (1, 4):
        for impairments_level in range(0, 3):
            futures.append(pool.submit(create_dataset, max_signals, impairments_level, ["am", "fm"]))
    for future in futures:
        if future.exception():
            print(future.exception())
