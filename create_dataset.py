# define dataset metadata, can override defaults
dataset_metadata = TorchSigDefaults().default_dataset_metadata

# optionally, apply impairments
impairments = Impairments(level=0)
burst_impairments = impairments.signal_transforms
whole_signal_impairments = impairments.dataset_transforms

# create the dataset
dataset = TorchSigIterableDataset(
    metadata=dataset_metadata,
    transforms=[
        whole_signal_impairments,
        Spectrogram(fft_size=dataset_metadata["fft_size"]),
    ],
    component_transforms=[burst_impairments],
)
# create a dataloader (reproducible)
dataloader = WorkerSeedingDataLoader(dataset, batch_size=2)

# save the dataset to disk
dataset_creator = DatasetCreator(
    dataset_length=20,
    dataloader=dataloader,
    root="./sample_dataset",
    overwrite=True,
    multithreading=False,
)
dataset_creator.create()
