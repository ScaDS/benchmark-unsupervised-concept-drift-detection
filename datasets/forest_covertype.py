from os import path

from river.datasets import base
from river import stream


class ForestCovertype(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=581_012,
            n_features=54,
            task=base.MULTI_CLF,
            filename="covtype.data",
        )
        self.full_path = path.join(directory_path, self.filename)

        # Define column names in the exact order of the CSV
        self.columns = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
            "Wilderness_Area1",
            "Wilderness_Area2",
            "Wilderness_Area3",
            "Wilderness_Area4",
        ] + [f"Soil_Type{i}" for i in range(1, 41)] + ["class"]

    def __iter__(self):
        return stream.iter_csv(
            self.full_path,
            target="class",
            fieldnames=self.columns,   # 👈 key change: provide header manually
            converters={
                **{name: float for name in self.columns[:10]},
                **{name: int for name in self.columns[10:]},
            }
        )
