import json
import os
import dask.array as da
import numpy as np

from spotiflow.data.spots2dt import Spots2DT
from spotiflow.augmentations.transforms3d import Crop3D

def main(training_data_dir: str, good_datasets: list):
    with open(training_data_dir, "r") as f:
        training_data = json.load(f)

    # Process training data
    imgs = []
    annotations = []
    for i, zarr_dataset in enumerate(training_data):
        if i in good_datasets:
            print(f"Processing dataset {i}...")
            zarr_path = os.path.join(zarr_dataset, "analysis/max_projections/maxz")

            # Process zarr
            img = da.from_zarr(zarr_path)
            img = img[:,0,0,:,:]
            print(img.shape)
            imgs.append(img)

            # Process annotations
            annotation = training_data[zarr_dataset]
            annotation = np.array(annotation)
            annotation = np.delete(annotation, 1, axis=1)  # remove z coordinate
            print(annotation)
            annotations.append(annotation)

    cropper = Crop3D(size=(16, 64, 64), point_priority=0.9)

    ds = Spots2DT(
        images = imgs,
        centers = annotations,
        cropper = cropper,
        defer_normalization=True,
        add_class_label=False,
    )

    item = ds[0]

if __name__ == "__main__":
    training_data_dir = "/Volumes/sgrolab/jennifer/predicty/training_data_mac.json"
    good_datasets = [8, 9, 10, 14, 16, 17, 18]
    main(training_data_dir, good_datasets)