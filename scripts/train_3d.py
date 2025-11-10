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


def load_zarr(zarr_path: str):
    try:
        dataset = da.from_zarr(zarr_path)
        if dataset.dtype.byteorder == '>':
            dataset = dataset.astype(dataset.dtype.newbyteorder('<'))
        print(f"Loaded dataset from {zarr_path} with shape {dataset.shape}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {zarr_path}: {e}")
        return None
    

def process_annotations(dataset: da.Array, annotations: np.ndarray, early_predict_frames: int):
    imgs =[]
    annots = []
    for annot in annotations:
        tp = annot[0] - early_predict_frames
        img = dataset[tp, 0]  # cells are channel 0
        imgs.append(img)
        annots.append(np.array([annot[1:]]))
    return imgs, annots


def create_dataset(training_data: list, early_predict_frames: int):
    all_imgs =[]
    all_annots = []
    for zarr_path, annotations in training_data:
        dataset = load_zarr(os.path.join(zarr_path, "0", "0"))
        if dataset is None:
            continue

        # Process annotations
        dataset_annotations = np.array(annotations) # tzyx
        imgs, annots = process_annotations(dataset, dataset_annotations, early_predict_frames)
        all_imgs.extend(imgs)
        all_annots.extend(annots)
    return all_imgs, all_annots


def main(data_dir: str, early_predict_frames: int = 1):
    
    with open(data_dir, "r") as f:
        data = json.load(f)

    data_list = list(data.items())

    cropper = Crop3D(size=(64, 256, 256), point_priority=1.0) # Always train/predict on crops containing spots

    aug_pipeline = Pipeline()
    aug_pipeline.add(FlipRot90(probability=0.5))
    aug_pipeline.add(RotationYX3D(angle=(-15, 15), probability=0.5, order=1))
    aug_pipeline.add(GaussianNoise3D(sigma=(0.0, 0.02), probability=0.85))
    aug_pipeline.add(IntensityScaleShift3D(scale=(0.8, 1.2), shift=(-0.1, 0.1), probability=0.85))

    # Process training data
    print("Loading training data...")
    train_imgs, train_annots = create_dataset(data_list[0:-3], early_predict_frames)

    train_data = Spots2DT( # Use 2DT class for 3D data with cropping
        images=train_imgs,
        centers=train_annots,
        cropper=cropper,
        defer_normalization=True,
        add_class_label=False,
        compute_flow=True,
        sigma=17,
        downsample_factors=(1, 2, 4, 8),
        augmenter=aug_pipeline,
    )
    print(f"Training data loaded (N={len(train_data.images)}).")

    # Process validation data
    print("Loading validation data...")
    val_imgs, val_annots = create_dataset(data_list[-3:], early_predict_frames)

    val_data = Spots2DT(
        images=val_imgs,
        centers=val_annots,
        cropper=cropper,
        defer_normalization=True,
        add_class_label=False,
        compute_flow=True,
        sigma=17,
        downsample_factors=(1, 2, 4, 8),
    )
    print(f"Validation data loaded (N={len(val_data.images)}).")

    # #Visualize a random crop
    # idx = np.random.randint(0, len(val_data))
    # idx = 8
    # print(f"Selected index: {idx}")

    # item = val_data[idx]
    # img = item["img"].squeeze().numpy() 
    # pts = item["pts"].numpy()
    # heatmap = item["heatmap_lv0"].squeeze().numpy()
    # flow = item["flow"].permute(1, 2, 3, 0).numpy()
    # print(f"Image shape: {img.shape}, pts shape: {pts.shape}, heatmap shape: {heatmap.shape}, flow shape: {flow.shape}")

    # viewer = napari.Viewer()
    # viewer.add_image(img, name="img")
    # viewer.add_points(pts, size=1, name="pts", symbol="disc", border_color="magenta", face_color="magenta")
    # viewer.add_image(heatmap, name="heatmap", colormap="magma", blending="additive", opacity=0.6)
    # napari.run()

    print("Instantiating new model...")
    model = Spotiflow(SpotiflowModelConfig(
        backbone="unet",
        in_channels=1,
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


if __name__ == "__main__":
    data_dir = "/groups/sgro/sgrolab/jennifer/predicty/training_data_server.json"
    main(data_dir, early_predict_frames=1)
