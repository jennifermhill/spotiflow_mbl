import os
import json
import dask.array as da
import numpy as np
import napari
import torch

from spotiflow.model import Spotiflow
from spotiflow.augmentations.transforms3d import Crop3D
from spotiflow.data.spots2dt import Spots2DT

def shift_annotations(annotations: list, shift_size: int = 32) -> np.ndarray:
    for annotation in annotations:
        annotation[0] -= shift_size
        if annotation[0] <= 0:
            annotation[0] = 0
    return annotations

def decuplicate_annotations(annotations: list, n_duplicates: int = 10) -> list:
    # For each annotation in the list of annotations, decuplicate with values in the zero index descending by 1 each
    dataset_annotations = []
    for annotation in annotations:
        dataset_annotations.append(annotation)
        for j in range(1, n_duplicates):
            duplicate = annotation.copy()
            duplicate[0] -= j
            if duplicate[0] < 0:
                break
            dataset_annotations.append(duplicate)

    return dataset_annotations

def main(training_data_dir: str, 
         pred_dataset: int,
         model_time: str,
         shift_forward: bool = False,
         decuplicate: bool = False):
    
    with open(training_data_dir, "r") as f:
        training_data = json.load(f)

    # Process training data
    imgs = []
    all_annotations = []
    annotations_tot = 0
    for i, zarr_dataset in enumerate(training_data):
        print(f"Processing dataset {i}...")
        zarr_path = os.path.join(zarr_dataset, "../analysis/max_projections/maxz")

        # Process zarr
        img = da.from_zarr(zarr_path)
        img = img[:,0,0,:,:]
        imgs.append(img)

        # Process annotations
        dataset_annotations = training_data[zarr_dataset]

        if shift_forward:
            dataset_annotations = shift_annotations(dataset_annotations, shift_size=48)

        if decuplicate:
            dataset_annotations = decuplicate_annotations(dataset_annotations, n_duplicates=10)

        dataset_annotations = np.array(dataset_annotations) # tzyx
        dataset_annotations = dataset_annotations[:,[0,2,3]]
        print(f"Number of annotations in dataset {i}: {dataset_annotations.shape[0]}")

        annotations_tot += dataset_annotations.shape[0]
        all_annotations.append(dataset_annotations)
    print(f"Total number of annotations across all datasets: {annotations_tot}")

    cropper = Crop3D(size=(64, 256, 256), point_priority=1.0)

    print("Loading validation data...")
    val_data = Spots2DT(
        images = imgs[-3:],
        centers = all_annotations[-3:],
        cropper = cropper,
        defer_normalization=True,
        add_class_label=False,
        compute_flow=True,
        sigma=17,
        downsample_factors=(1, 2, 4, 8)
    )
    print(f"Validation data loaded (N={len(val_data.images)}).")

    print("Loading validation image...")
    idx = 0
    item = val_data[idx]
    val_crop = item["img"][:, :]
    print(f"Image shape is: {val_crop.shape}")
    heatmap = item["heatmap_lv0"][:, 0:45].squeeze().numpy()

    # viewer = napari.Viewer()
    # viewer.add_image(val_crop, name="img")
    # viewer.add_image(heatmap, name="heatmap")
    # napari.run()

    # load pre-trained model
    print("Loading model...")
    model = Spotiflow.from_folder("/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-" + model_time, map_location="cuda")
    model.to(torch.device("cuda"))

    #n_tiles = tuple(max(s//g, 1) for s, g in zip(val_imgs.shape, (16, 256, 256)))
    #print(n_tiles)
    print("Predicting volume...")

    #convert val_crop to numpy
    val_crop = val_crop.squeeze(0).numpy()

    # Compare usual input to current input
    val_imgs = imgs[pred_dataset].astype(np.float32).compute()
    print(f"Image shape is: {val_imgs.shape}, dtype: {val_imgs.dtype}")
    print(f"Crop shape is: {val_crop.shape}, dtype: {val_crop.dtype}")

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
    viewer.add_image(details.heatmap, name="heatmap")
    viewer.add_image((details.flow+1)*0.5, name="flow")
    napari.run()

if __name__ == "__main__":
    training_data_dir = "/groups/sgro/sgrolab/jennifer/predicty/training_data_server.json"
    main(training_data_dir, 
         pred_dataset=16,
         model_time = "20250903_1715",
         shift_forward=True,
         decuplicate=True,
         )