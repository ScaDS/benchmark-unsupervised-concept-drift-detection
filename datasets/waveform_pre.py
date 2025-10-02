from os import path

from river.datasets import base
from river import stream


class WaveformPre(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=15000,
            n_features=21,
            task=base.MULTI_CLF,
            filename="waveform.csv",
        )
        self.full_path = path.join(directory_path, self.filename)
        self.drifts=[]
        with open("datasets/files/waveform_drifts.csv", 'r') as f:
            self.drifts = [int(line.strip()) for line in f]

    def __iter__(self):
        converters = {f"feature_{i}": float for i in range(0, 21)}
        converters["class"] = int
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters=converters,
        )

