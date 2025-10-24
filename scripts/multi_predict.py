import os
import json
import dask.array as da
import numpy as np
import napari
import torch
import pandas as pd
import tifffile as tiff

from spotiflow.data.spots2dt import Spots2DT
from spotiflow.model import Spotiflow
from spotiflow.utils.matching import points_matching

def predict_annotation_site(img, model, annotation, prob_thresh=0.3):
    # predict on entire data set for a single annotated site
    spots, details = model.predict(
        img,
        subpix=True,
        min_distance=75,
        prob_thresh=prob_thresh,
        #n_tiles=n_tiles, # change if you run out of memory
        device="cuda",
    )

    

def main(data_dir: str, model_time: str, prob_thresh: float):
    '''Predict on all annotated sites in data_dir using model specified by model_time.
    
    Args:
        data_dir (str): Directory containing training and validation annotations. Training annotations are at the last time point of the aggregation.
                        Validation annotations are at the earliest recognizable aggregation time point.
        model_time (str): Timestamp of the model to use for prediction.
        shift_forward (int): Number of frames training annotations were shifted forward during training (to be matched when predicting).
    '''
    print(f"Predicting with model {model_time}")

    training_annotations = os.path.join(data_dir, "training_data_server.json")
    validation_annotations = os.path.join(data_dir, "validation_data_server.json")

    with open(training_annotations, "r") as f:
        training_annotations = json.load(f)

    with open(validation_annotations, "r") as f:
        validation_annotations = json.load(f)

    # Process data and annotations
    imgs = []
    train_annotations = []
    val_annotations = []
    for i, zarr_dataset in enumerate(training_annotations):
        print(f"Processing dataset {i}...")
        # zarr_path = os.path.join(zarr_dataset, "analysis_mbl/max_projections/maxz")
        zarr_path = os.path.join(zarr_dataset, "analysis/max_projections/maxz")

        # Process zarr
        try:
            img = da.from_zarr(zarr_path)
        except Exception as e:
            print(f"Skipping dataset {i}: {e}")
            continue
        img = img[:,0,0,:,:]
        # img = img[:,[0,3],:,:].transpose(0,2,3,1) 
        
        imgs.append(img)

        # Process training annotations
        train_annotation = training_annotations[zarr_dataset]
        train_annotation = np.array(train_annotation) # tzyx 
        # TODO: keep z coordinate to compare performance on low vs high aggregations
        train_annotation = train_annotation[:,[0,2,3]] # txy
        train_annotations.append(train_annotation)

        # Process validation annotations
        val_annotation = validation_annotations[zarr_dataset]
        val_annotation = np.array(val_annotation) # tzyx
        val_annotation = val_annotation[:,[0,2,3]] # txy
        val_annotations.append(val_annotation)

        breakpoint()

    # load pre-trained model
    print("Loading model...")
    model_path = "/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-" + model_time
    model = Spotiflow.from_folder(model_path, map_location="cuda")
    model.to(torch.device("cuda"))

    # Create dataset object
    print("Loading validation data...")
    val_data = Spots2DT(
        images = imgs[-3:],
        centers = val_annotations[-3:],
        add_class_label=False,
        sigma=17,
        downsample_factors=(1, 2, 4, 8)
    )
    print(f"Validation data loaded (N={len(val_data.images)}).")

    breakpoint()
    # Output should be a dataframe of true positives, false positives, false negatives for each validation dataset
    metrics = pd.DataFrame(columns=["dataset", "TP", "FP", "FN"])
    for i, dataset in enumerate(val_annotations):
        tps = 0
        fps = 0
        fns = 0
        for j, val_annot in enumerate(dataset):
            img = val_data.get_predict_item(dataset_id=i, annotation_id=j)
            tp, fp, fn = predict_annotation_site(img, model, val_annot, prob_thresh=prob_thresh)
            tps += tp
            fps += fp
            fns += fn
        metrics = pd.concat([metrics, pd.DataFrame({"dataset": [dataset], "TP": [tps], "FP": [fps], "FN": [fns]})], ignore_index=True)

if __name__ == "__main__":
    data_dir = "/groups/sgro/sgrolab/jennifer/predicty/"
    model_time = "20250903_1715"
    prob_thresh = 32

    main(data_dir, model_time, prob_thresh)
