import json
import os
import dask.array as da
import numpy as np
import napari
import datetime
import lightning.pytorch as pl

from spotiflow.data.spots2dt import Spots2DT
from spotiflow.augmentations.transforms3d import Crop3D, IntensityScaleShift3D, GaussianNoise3D, RotationYX3D
from spotiflow.model import Spotiflow, SpotiflowModelConfig, SpotiflowTrainingConfig
from spotiflow.model.trainer import SpotiflowModelCheckpoint
from spotiflow.augmentations.pipeline import Pipeline
from spotiflow.augmentations.transforms import FlipRot90


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
         good_datasets: list, 
         decuplicate: bool = False,
         shift_forward: bool = False,
         ):
    
    with open(training_data_dir, "r") as f:
        training_data = json.load(f)

    # Process training data
    imgs = []
    all_annotations = []
    annotations_tot = 0
    for i, zarr_dataset in enumerate(training_data):
        if i in good_datasets:
            print(f"Processing dataset {i}...")
            zarr_path = os.path.join(zarr_dataset, "analysis_mbl/max_projections/maxz")

            # Process zarr
            try:
                img = da.from_zarr(zarr_path)
            except Exception as e:
                print(f"Skipping dataset {i}: {e}")
                continue
            img = img[:,[0,3,4],:,:].transpose(1,0,2,3)

            imgs.append(img)

            # Process annotations
            dataset_annotations = training_data[zarr_dataset]

            if shift_forward:
                dataset_annotations = shift_annotations(dataset_annotations, shift_size=32)

            if decuplicate:
                dataset_annotations = decuplicate_annotations(dataset_annotations, n_duplicates=10)

            dataset_annotations = np.array(dataset_annotations) # tzyx
            dataset_annotations = dataset_annotations[:,[0,2,3]]
            print(f"Number of annotations in dataset {i}: {dataset_annotations.shape[0]}")

            annotations_tot += dataset_annotations.shape[0]
            all_annotations.append(dataset_annotations)
    print(f"Total number of annotations across all datasets: {annotations_tot}")

    cropper = Crop3D(size=(64, 256, 256), point_priority=1.0)

    aug_pipeline = Pipeline()
    aug_pipeline.add(FlipRot90(probability=0.5))
    aug_pipeline.add(RotationYX3D(angle=(-15, 15), probability=0.5, order=1))
    aug_pipeline.add(GaussianNoise3D(sigma=(0.0, 0.02), probability=0.85))
    aug_pipeline.add(IntensityScaleShift3D(scale=(0.8, 1.2), shift=(-0.1, 0.1), probability=0.85))

    print("Loading training data...")
    train_data = Spots2DT(
        images = imgs[0:-3],
        centers = all_annotations[0:-3],
        cropper = cropper,
        defer_normalization=True,
        add_class_label=False,
        compute_flow=True,
        sigma=17,
        downsample_factors=(1, 2, 4, 8),
        augmenter=aug_pipeline,
    )
    print(f"Training data loaded (N={len(train_data.images)}).")
    
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

    #Visualize a random crop
    # idx = np.random.randint(0, len(val_data))
    # print(idx)

    # item = val_data[idx]
    # img = item["img"].squeeze().numpy() 
    # pts = item["pts"].numpy()
    # heatmap = item["heatmap_lv0"].squeeze().numpy()
    # flow = item["flow"].permute(1, 2, 3, 0).numpy()
    # print(f"Image shape: {img.shape}, pts shape: {pts.shape}, heatmap shape: {heatmap.shape}, flow shape: {flow.shape}")

    # viewer = napari.Viewer()
    # # viewer.add_image(imgs[-1])
    # #viewer.add_points(annotations[17], size=10, name="pts", symbol="disc", border_color="red", face_color="red")
    # viewer.add_image(img, name="img")
    # # viewer.add_points(spots, size=1, name="pts", symbol="disc", border_color="magenta", face_color="magenta")
    # viewer.add_image(heatmap, name="heatmap")
    # # viewer.add_image((details.flow+1)*0.5, name="flow")
    # napari.run()

    # 1/0

    print("Instantiating new model...")
    model = Spotiflow(SpotiflowModelConfig(
        backbone="unet",
        in_channels=3,
        out_channels=1,
        sigma=17,
        is_3d=True,
    ))

    logger = pl.loggers.TensorBoardLogger(save_dir="./logs", name=f"spotiflow-{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
    output_dir = "./outputs/spotiflow-" + datetime.datetime.now().strftime('%Y%m%d_%H%M')
    train_config = SpotiflowTrainingConfig(
        crop_size=256,
        batch_size=8,
        crop_size_depth=64,
        num_epochs=1000,
        )

    callbacks = [
        SpotiflowModelCheckpoint(
            logdir=output_dir,
            train_config=train_config,
            monitor="val_loss",
        )
    ]

    print("Launching training...")
    model.fit_dataset(
        train_data,
        val_data,
        train_config=train_config,
        logger=logger,
        default_root_dir=output_dir,
        callbacks=callbacks,
    )

    print("Done!")

    # print("Predicting validation dataset")
    # print(f"Validation image shape = {val_data[0]['img'].shape}")
    # pred, details = model.predict(val_data[0]['img'].squeeze().numpy())
    # print(f"Spots shape: {pred.shape}")

    # viewer = napari.Viewer()
    # # # viewer.add_image(imgs[-1])
    # # # viewer.add_points(annotations[-1], size=10, name="pts", symbol="disc", border_color="red", face_color="red")
    # # viewer.add_image(val_data[0]["img"], name="img")
    # # viewer.add_points(pred[0], size=1, name="pts", symbol="disc", border_color="red", face_color="red")
    # # viewer.add_image(pred[1]["heatmap_lv0"], name="heatmap")
    # # viewer.add_image((pred[1]["flow"]+1)*0.5, name="flow")
    # viewer.add_image(val_data[0]["img"], name="img")
    # viewer.add_image(details.heatmap, name="hmap", contrast_limits=(0,1), colormap="magma")
    # viewer.add_image((details.flow+1)*0.5, name="flow")
    # viewer.add_points(pred, name="pts")
    # viewer.add_points(annotations[-1], size=10, name="true_pts", symbol="disc", border_color="red", face_color="red")
    # napari.run()


if __name__ == "__main__":
    training_data_dir = "/groups/sgro/sgrolab/jennifer/predicty/training_data_server.json"
    good_datasets = range(0, 19)
    main(training_data_dir, 
         good_datasets, 
         decuplicate=True,
         shift_forward=True,
         )