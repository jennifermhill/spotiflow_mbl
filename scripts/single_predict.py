import os
import json
import dask.array as da
import numpy as np
import napari
import torch

from spotiflow.model import Spotiflow
from spotiflow.augmentations.transforms3d import Crop3D
from spotiflow.data.spots2dt import Spots2DT

def main(
        zarr_dataset: str,
        model_time: str
        ) -> None:

    # Load input image
    zarr_path = os.path.join(zarr_dataset, "../analysis/max_projections/maxz")
    img = da.from_zarr(zarr_path)
    img = img[:,1,0,:,:]
    # Downsample for testing
    img = img[:,::4,::4]

    annot = np.array([[33, 144, 144]])  # Choose the center

    # Make data cropper and loader
    cropper = Crop3D(size=(64, 256, 256), point_priority=1.0)

    data_loader = Spots2DT(
        images=[img],
        centers=[annot],
        cropper=cropper,
        defer_normalization=True,
        add_class_label=False,
        compute_flow=True,
        sigma=17,
        downsample_factors=(1, 2, 4, 8)
    )

    # Check cropped image
    print("Loading image...")
    idx = 0
    item = data_loader[idx]
    val_crop = item["img"][:, :]
    print(f"Image shape is: {val_crop.shape}")
    heatmap = item["heatmap_lv0"][:, :].squeeze().numpy()

    viewer = napari.Viewer()
    viewer.add_image(val_crop, name="img")
    viewer.add_image(heatmap, name="heatmap", colormap="magma", opacity=0.6)
    napari.run()

    # Load model
    print("Loading model...")
    model = Spotiflow.from_folder("/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-" + model_time, map_location="cuda")
    model.to(torch.device("cuda"))

    # Make predictions
    print("Making predictions...")

    val_crop = val_crop.squeeze(0).numpy()

    spots, details = model.predict(
        val_crop,
        subpix=True,
        min_distance=75,
        #n_tiles=n_tiles, # change if you run out of memory
        device="cuda",
    )

    viewer = napari.Viewer()
    # viewer.add_image(imgs[-1])
    #viewer.add_points(all_annotations[pred_dataset], size=20, name="pts", symbol="disc", border_color="red", face_color="red")
    viewer.add_image(item["img"], name="img")
    viewer.add_image(heatmap, name="heatmap")
    viewer.add_points(spots, size=20, name="pts", symbol="disc", border_color="blue", face_color="blue")
    viewer.add_image(details.heatmap, name="heatmap", colormap="magma", opacity=0.6)
    #viewer.add_image((details.flow+1)*0.5, name="flow")
    napari.run()

if __name__ == "__main__":
    zarr_dataset = "/groups/sgro/sgrolab/jennifer/cryolite/John/102325_nc281-spiAmSG_cryolite_20Xwi_timelapse001"
    model_time = "20250903_1715"
    main(zarr_dataset, model_time)
