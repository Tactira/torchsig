import glob
import os
import shutil
from sigmf import SigMFFile
from torchsig.datasets.datasets import StaticTorchSigDataset

SAMPLE_RATE = int(20.48e6)
CF = 0
P = "/scratch/sdrdata/hdf5/*"

for d in glob.glob(P):
    b = os.path.basename(d)
    o = os.path.join("/scratch/sdrdata/sigmf", b)
    if os.path.exists(o):
        shutil.rmtree(o)
    os.mkdir(o)

    dataset = StaticTorchSigDataset(d)

    for i, sig in enumerate(dataset):
        data = sig.data
        meta = sig["metadata"]
        class_names = "-".join([c.class_name for c in sig.component_signals])
        data_file = os.path.join(o, f"{class_names}-{i}.sigmf-data")
        data.tofile(data_file)
        print(i, data_file)
        print(sig, data.shape, data.dtype)

        sf = SigMFFile(
            data_file=data_file,
            global_info={
                SigMFFile.DATATYPE_KEY: "cf32_le",
                SigMFFile.SAMPLE_RATE_KEY: SAMPLE_RATE,
            },
        )

        sf.add_capture(
            0,
            metadata={
                SigMFFile.FREQUENCY_KEY: CF,
            },
        )
        print("x" * 80)
        print(sig)

        for c in sig.component_signals:
            cm = c.metadata
            ctr = float(cm["center_freq"]) + CF
            print(cm)
            sf.add_annotation(
                int(cm["start_in_samples"]),
                metadata={
                    SigMFFile.FHI_KEY: float(cm["_upper_frequency"]),
                    SigMFFile.FLO_KEY: float(cm["_lower_frequency"]),
                    SigMFFile.LENGTH_INDEX_KEY: int(cm["duration_in_samples"]),
                    SigMFFile.LABEL_KEY: cm["class_name"],
                },
            )

        # f Write to disk
        meta_file = os.path.join(o, f"{class_names}-{i}.sigmf-meta")
        sf.tofile(meta_file, overwrite=True)
