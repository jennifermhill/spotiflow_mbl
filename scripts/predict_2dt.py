import os
import json
import dask.array as da
import numpy as np
import napari
import torch

from spotiflow.model import Spotiflow

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
            imgs.append(img)

            # Process annotations
            annotation = training_data[zarr_dataset]
            annotation = np.array(annotation) # tzyx
            annotation = annotation[:,[0,2,3]]
            # np.delete(annotation, 1, axis=1)  # remove z coordinate
            # annotation = annotation[:,[0,2,1]]  # txy

            annotations.append(annotation)

    # load pre-trained model
    print("Loading model...")
    model = Spotiflow.from_folder("/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-20250902_1536", map_location="cuda")
    model.to(torch.device("cuda"))
    print("Loading validation image...")
    val_imgs = imgs[-1].astype(np.float32).compute()

    print(f"Image shape is: {val_imgs.shape}")

    #n_tiles = tuple(max(s//g, 1) for s, g in zip(val_imgs.shape, (16, 256, 256)))
    #print(n_tiles)
    print("Predicting volume...")

    spots, details = model.predict(
        val_imgs,
        subpix=True,
        #n_tiles=n_tiles, # change if you run out of memory
        device="cuda",
    )

    viewer = napari.Viewer()
    # viewer.add_image(imgs[-1])
    viewer.add_points(annotations[-1], size=10, name="pts", symbol="disc", border_color="red", face_color="red")
    viewer.add_image(val_imgs, name="img")
    viewer.add_points(spots, size=1, name="pts", symbol="disc", border_color="magenta", face_color="magenta")
    viewer.add_image(details.heatmap, name="heatmap")
    viewer.add_image((details.flow+1)*0.5, name="flow")
    napari.run()

if __name__ == "__main__":
    training_data_dir = "/groups/sgro/sgrolab/jennifer/predicty/training_data_server.json"
    good_datasets = range(0, 19)
    main(training_data_dir, good_datasets)