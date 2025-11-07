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


def extract_matched_pairs(stats, annotations, spots, shift_forward):
    '''Extract matched pairs of annotations and predicted spots based on matching statistics.
    Args:
        stats: Matching statistics object containing indices of matched pairs.
        annotations: Array of annotation coordinates (t, z, x, y).
        spots: Array of predicted spot coordinates (t, x, y).
        shift_forward: Integer value to adjust time coordinate of annotations.
    Returns:
        Array of matched pairs with shape (N, 2, 4) where N is the number of matched pairs,
        and each pair contains the annotation and corresponding predicted spot coordinates 
        (t, x, y, z).
    '''
    pairs_ids = np.array(stats.matched_pairs)
    pairs = np.zeros((pairs_ids.shape[0], 2, 4), dtype=np.float32)
    pairs[:, 0, :] = annotations[pairs_ids[:, 0]][:, [0, 2, 3, 1]] # reorder to t,x,y,z
    pairs[:, 1, :-1] = spots[pairs_ids[:, 1]] # add t,x,y
    pairs[:, 1, 3] = pairs[:, 0, 3] # add z from annotations
    pairs[:, 0, 0] += shift_forward # adjust time coordinate of annotations back
    return pairs

def main(
        data_dir: str, 
        model_time: str, 
        shift_forward: int = 0, 
        window_size: int = 32,
        prob_thresh: float = 0.3
        ):
    '''Predict on all annotated sites in data_dir using model specified by model_time.
    
    Args:
        data_dir (str): Directory containing training and validation annotations. Training annotations are at the last time point of the aggregation.
                        Validation annotations are at the earliest recognizable aggregation time point.
        model_time (str): Timestamp of the model to use for prediction.
        shift_forward (int): Number of frames training annotations were shifted forward during training (to be matched when predicting). Defaults to 0.
        window_size (int): Number of timepoints to use for each prediction window. Defaults to 32.
        prob_thresh (float): Probability threshold for spot detection. Defaults to 0.3.
    '''
    print(f"Predicting with model {model_time}")

    annotations_path = os.path.join(data_dir, "training_data_server.json")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    annotations = list(annotations.items())[-3:]  # last 3 are validation datasets

    # Load pre-trained model
    print("Loading model...")
    model_path = "/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-" + model_time
    model = Spotiflow.from_folder(model_path, map_location="cuda")
    model.to(torch.device("cuda"))

    # Initialize metrics dataframe
    metrics = pd.DataFrame(columns=["dataset", "TP", "FP", "FN"])
    dist_error = []
    early_detect_frames_list = []
    surface_aggs = 0
    matched_pairs_data = []

    # Predict on each validation dataset
    for i, zarr_dataset in enumerate(annotations):
        print(f"Processing dataset {i}...")

        zarr_dataset_path = zarr_dataset[0]

        # Process annotations
        ds_annotations = zarr_dataset[1]
        ds_annotations = np.array(ds_annotations) # tzyx
        # ds_annotations = ds_annotations[:,[0,2,3]] # txy
        # subtract shift_forward if training annotations were shifted
        ds_annotations[:,0] -= shift_forward

        # Load full dataset
        # zarr_path = os.path.join(zarr_dataset, "analysis_mbl/max_projections/maxz")
        zarr_path = os.path.join(zarr_dataset_path, "analysis/max_projections/maxz")

        # Process zarr
        try:
            img = da.from_zarr(zarr_path)
        except Exception as e:
            print(f"Skipping dataset {i}: {e}")
            continue
        img = img[:,0,0,:,:]
        # img = img[:,[0,3],:,:].transpose(0,2,3,1) 

        # TODO: get size in Z from metadata

        tps = 0
        fps = 0
        all_spots = []
        all_matched_annotations = set()

        # Generate rolling windows of 32 timepoints to predict on
        n_windows = img.shape[0] - window_size - 9 # don't use windows from last 10 timepoints since there are no annotations beyond the end

        for w in range(0, n_windows, 5):
            print(f"Predicting window {w+1}/{n_windows}...")
            img_window = img[w:w+window_size]
            img_window = img_window.astype(np.float32).compute()
            spots, details = model.predict(
                img_window,
                subpix=True,
                min_distance=75,
                prob_thresh=prob_thresh,
                #n_tiles=n_tiles, # change if you run out of memory
                device="cuda",
            )

            if len(spots) == 0:
                continue
            
            new_spots = []
            for spot in spots:
                spot[0] += w  # shift time coordinate back to full dataset
                
                # Check if spot has already been predicted
                if len(all_spots) > 0:
                    duplicate_stats = points_matching(
                        p1=np.array(all_spots),
                        p2=np.array([spot]),
                        cutoff_distance=200,
                        eps=1e-8,
                    )
                    if duplicate_stats.tp > 0:
                        print("Spot already predicted, skipping...")
                        continue
                    else:
                        new_spots.append(spot)
                else:
                    new_spots.append(spot)

            if len(new_spots) > 0:
                new_spots = np.array(new_spots)
                # Check if new spots match any annotations
                stats = points_matching(
                    p1=ds_annotations[:, [0, 2, 3]],  # txy
                    p2=new_spots,
                    cutoff_distance=200,
                    eps=1e-8,
                )
                print(f"Found {stats.tp} true positives for annotations.")
                print(f"Found {stats.fp} false positives for annotations.")

                if len(stats.matched_pairs) > 0:
                    dist_error.extend(stats.dist.tolist()) # calculated as error in t, x, y where t has been shifted forward
                    matched_pairs = extract_matched_pairs(stats, ds_annotations, new_spots, shift_forward)
                    early_detect_frames = (matched_pairs[:, 0, 0] - matched_pairs[:, 1, 0]).tolist()
                    early_detect_frames_list.extend(early_detect_frames)
                    for pair in matched_pairs:
                        if pair[0, 3] > 60:  # arbitrary threshold for high surface aggregation
                            surface_aggs += 1
                    matched_pairs_data.append({
                        "dataset": zarr_dataset_path,
                        "window": w,
                        "matched_pairs": matched_pairs.tolist(),
                        "distance_errors": stats.dist.tolist(),
                        "mean_distance_error": stats.mean_dist,
                        "early_detect_frames": early_detect_frames,
                        "mean_early_detect_frames": np.mean(early_detect_frames),
                        "surface_aggregations": surface_aggs,
                    })
                    # Add matched annotations to set
                    matched_annotation_indices = [pair[0] for pair in stats.matched_pairs]
                    all_matched_annotations.update(matched_annotation_indices)

                tps += stats.tp
                fps += stats.fp
                all_spots.extend(spot for spot in new_spots)

        total_annotations = ds_annotations.shape[0]
        unique_matched_annotations = len(all_matched_annotations)
        fns = total_annotations - unique_matched_annotations

        print(f"Dataset {i} - TP: {tps}, FP: {fps}, FN: {fns}")
        metrics = pd.concat([metrics, pd.DataFrame({"dataset": [zarr_dataset_path], "TP": [tps], "FP": [fps], "FN": [fns]})], ignore_index=True)

    # Save metrics to CSV
    avg_dist_error = np.mean(dist_error) if len(dist_error) > 0 else 0
    print(f"Average distance error across all datasets: {avg_dist_error}")
    avg_early_detect = np.mean(early_detect_frames_list) if len(early_detect_frames_list) > 0 else 0
    print(f"Average early detection frames across all datasets: {avg_early_detect}")
    print(f"Number of surface aggregations detected: {surface_aggs}")
    # Add average distance error to metrics dataframe
    metrics["avg_distance_error"] = avg_dist_error
    metrics["avg_early_detection_frames"] = avg_early_detect
    metrics["surface_aggregations"] = surface_aggs  
    metrics_path = os.path.join(model_path, f"prediction_metrics_{window_size}ws_{prob_thresh}pt.csv")
    metrics.to_csv(metrics_path, index=False)

    # Save matched pairs data to JSON
    matched_pairs_path = os.path.join(model_path, f"matched_pairs_{window_size}ws_{prob_thresh}pt.json")
    with open(matched_pairs_path, "w") as f:
        json.dump(matched_pairs_data, f, indent=4)

if __name__ == "__main__":
    data_dir = "/groups/sgro/sgrolab/jennifer/predicty/"
    model_time = "20250903_2252"
    shift_forward = 48
    window_size = 50
    prob_thresh = 0.5

    main(data_dir, model_time, shift_forward, window_size, prob_thresh)
