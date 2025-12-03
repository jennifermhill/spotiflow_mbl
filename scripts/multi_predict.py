import os
import json
import numpy as np
import napari
import torch
import pandas as pd

from spotiflow.model import Spotiflow
from spotiflow.utils.matching import points_matching
from spotiflow.utils.mbl_utils import crop_manual, load_zarr, match_previous_window, predict_heatmap


def duplicate_points(points, nb_timepoints):
    """Duplicate points across timepoints."""
    duplicated = []
    for t in range(nb_timepoints):
        for point in points:
            duplicated.append([t, point[1], point[2]])  # t,y,x
    return np.array(duplicated)


def main(
        data_dir: str, 
        model_time: str, 
        shift_forward: int = 0, 
        window_size: int = 32,
        prob_thresh: float = 0.3,
        predict_z: bool = False,
        z_model_time: str = None,
        ):
    '''Predict on all annotated sites in data_dir using model specified by model_time.
    
    Args:
        data_dir (str): Directory containing training and validation annotations. Training annotations are at the last time point of the aggregation.
                        Validation annotations are at the earliest recognizable aggregation time point.
        model_time (str): Timestamp of the model to use for prediction.
        shift_forward (int): Number of frames training annotations were shifted forward during training (to be matched when predicting). Defaults to 0.
        window_size (int): Number of timepoints to use for each prediction window. Defaults to 32.
        prob_thresh (float): Probability threshold for spot detection. Defaults to 0.3.
        predict_z (bool): Whether to use 3D model to predict z positions of detected spots. Defaults to False (2D+t prediction only).
        z_model_time (str): Timestamp of the 3D model to use for predicting z positions if predict_z is True. Defaults to None.
    '''
    print(f"Predicting with 2dt model {model_time}")
    print(f"Predicting with 3d model {z_model_time}" if predict_z else "Not predicting z positions")

    annotations_path = os.path.join(data_dir, "training_data_server.json")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    annotations = list(annotations.items())[-1:]  # last 3 are validation datasets #real time annots

    # Load pre-trained model
    print("Loading 2dt model...")
    model_path = "/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-" + model_time
    model = Spotiflow.from_folder(model_path, map_location="cuda")
    model.to(torch.device("cuda"))

    if predict_z:
        print("Loading 3d model for z prediction...")
        model_3d_path = "/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-" + z_model_time
        model_3d = Spotiflow.from_folder(model_3d_path, map_location="cuda")
        model_3d.to(torch.device("cuda"))

    # Initialize metrics dataframe
    spot_stats_list = []

    # Predict on each validation dataset
    for i, zarr_dataset in enumerate(annotations):
        print(f"Processing dataset {i}...")

        zarr_dataset_path = zarr_dataset[0]

        # Process annotations
        gt_annotations = zarr_dataset[1] # real time annots
        gt_annotations = np.array(gt_annotations) # tzyx

        # Load full dataset
        zarr_path = os.path.join(zarr_dataset_path, "../analysis/max_projections/maxz")

        # Process zarr
        dataset = load_zarr(zarr_path)  # tc(maxz)yx
        dataset = dataset[:,0,0,:,:]

        if predict_z:
            dataset_3d = load_zarr(os.path.join(zarr_dataset_path, "0", "0"))  # tczyx
            dataset_3d = dataset_3d[:,0,:,:,:]

        tps = 0
        fps = 0
        all_spots = []
        all_matched_annotations = set()

        # Duplicate annotations across timepoints for visualization
        duplicate_gt_annotations = duplicate_points(gt_annotations[:, [0,2,3]], nb_timepoints=dataset.shape[0])

        # Generate rolling windows of window_size to predict on
        final_tp = gt_annotations[:,0].max()
        n_windows = max(1, final_tp - window_size) # don't use windows from beyond the last annotation

        for w in range(0, n_windows, 5):
            print(f"Predicting window {w+1}/{n_windows}...")
            img_window = dataset[w:w+window_size]
            img_window = img_window.astype(np.float32).compute()

            spots, probs, details = predict_heatmap(img_window, model, prob_thresh, num_peaks=8)

            if len(spots) > 0:
                ds_annotations = gt_annotations.copy()
                ds_annotations[:, 0] += shift_forward  # forward shifted annots for matching

                spots[:, 0] += w  # shift time coordinate back to full dataset

                stats = points_matching(
                    p1=ds_annotations[:, [0, 2, 3]],  # tyx # forward shifted annots
                    p2=spots,
                    cutoff_distance=200,
                    eps=1e-8,
                )
                print(f"Found {stats.tp} true positives for annotations.")
                print(f"Found {stats.fp} false positives for annotations.")


                spot_data_list = []
                for j, spot in enumerate(spots):
                    # Find if this spot is a TP
                    matched_pair_idx = next((idx for idx, pair in enumerate(stats.matched_pairs) if pair[1] == j), None)
                    is_tp = matched_pair_idx is not None

                    if len(all_spots) > 0:
                        # Match with spots from previous window
                        prev_spot_data_list = spot_stats_list[-1]
                        prev_spots = prev_spot_data_list[["spot_id", "spot_t", "spot_y", "spot_x"]].to_numpy()

                        curr_spot_idx = match_previous_window(spot, prev_spots)
                        if curr_spot_idx is None:
                            curr_spot_idx = len(all_spots) + j # assign new index
                    else:
                        curr_spot_idx = j

                    # Build spot data dictionary
                    spot_data = {
                        "dataset": zarr_dataset_path,
                        "frame_window": f"{w}-{w+window_size}",
                        "spot_id": curr_spot_idx,
                        "spot_t": int(spot[0]), # real time (corrected to full dataset)
                        "spot_y": int(spot[1]),
                        "spot_x": int(spot[2]),
                        "prob": float(probs[j]),
                        "TP/FP": "TP" if is_tp else "FP"
                    }

                    if is_tp:
                        # Get the ground truth annotation index for this matched spot
                        gt_idx = stats.matched_pairs[matched_pair_idx][0]
                        gt_annotation = gt_annotations[gt_idx]

                        # Add gt_annotation to list of all matched annotations
                        all_matched_annotations.add(gt_idx)
                        
                        spot_data["early_detect"] = int(gt_annotation[0] - (w + window_size)) # real time annot - window end
                        spot_data["dist_err_xy"] = float(np.linalg.norm(gt_annotation[[3,2]] - spot[[2,1]]))

                        if predict_z:
                            print(f"Predicting z position for TP {j}...")
                            img_3d = dataset_3d[w+window_size]  # use end of window for z prediction
                            annot_3d = gt_annotation[-3:]  # z,y,x

                            # Crop image and annotation manually
                            cropped_img, cropped_annot = crop_manual(img_3d, annot_3d, crop_size=(64, 128, 128))
                            cropped_img = cropped_img.astype(np.float32).compute()

                            # Predict z position
                            spots_3d, _, _ = predict_heatmap(cropped_img, model_3d, prob_thresh, num_peaks=1)

                            stats_3d = points_matching(
                                p1=np.array([cropped_annot]),
                                p2=np.array(spots_3d),
                                cutoff_distance=50,
                                eps=1e-8,
                            )

                            if stats_3d.tp > 0:
                                assert len(stats_3d.matched_pairs) == 1, "Expected exactly one matched pair"
                                print(f"TP {j} z position correctly predicted!")
                                spot_data["spot_z"] = int(spots_3d[0][0])
                                spot_data["dist_err_z"] = float(abs(cropped_annot[0] - spots_3d[0][0]))
                            else:
                                print(f"TP {j} z position incorrectly predicted :(")
                                spot_data["spot_z"] = np.nan
                                spot_data["dist_err_z"] = np.nan

                        else:
                            spot_data["spot_z"] = np.nan
                            spot_data["dist_err_z"] = np.nan
                    else:
                        spot_data["early_detect"] = np.nan
                        spot_data["dist_err_xy"] = np.nan
                    
                    spot_data_list.append(spot_data)

                spot_stats_list.append(pd.DataFrame(spot_data_list))

                all_spots.extend(spots)
                tps += stats.tp
                fps += stats.fp


            # Duplicate spots across timepoints for visualization
            duplicate_spots = duplicate_points(spots, nb_timepoints=dataset.shape[0])

            viewer = napari.Viewer()
            viewer.add_image(dataset, name="img")
            viewer.add_image(details.heatmap, name="heatmap", colormap="magma", opacity=0.6, translate=(w,0,0))
            viewer.add_points(list(duplicate_gt_annotations), size=20, name="annots", symbol="disc", border_color="red", face_color="red")
            viewer.add_points(list(duplicate_spots), size=20, name="spots", symbol="disc", border_color="blue", face_color="blue")
            # viewer.add_image((details.flow+1)*0.5, name="flow")
            napari.run()

        total_annotations = ds_annotations.shape[0]
        unique_matched_annotations = len(all_matched_annotations)
        fns = total_annotations - unique_matched_annotations

        print(f"Dataset {i} - TP: {tps}, FP: {fps}, FN: {fns}")
        
    print(f"spot_stats_list: {spot_stats_list}")
    spot_stats = pd.concat(spot_stats_list, ignore_index=True)

    # Save stats to CSV
    spot_stats_path = os.path.join(model_path, f"spot_stats_{window_size}ws_{prob_thresh}pt.csv")
    
    # More robust CSV writing with explicit flushing
    try:
        with open(spot_stats_path, 'w', newline='') as f:
            spot_stats.to_csv(f, index=False)
            f.flush()  # Ensure buffer is written
            os.fsync(f.fileno())  # Force write to disk
        print(f"Successfully saved {len(spot_stats)} rows to {spot_stats_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        # Fallback: try saving with a different method
        spot_stats.to_csv(spot_stats_path, index=False, mode='w')
        print(f"Saved using fallback method")

    # Verify the saved file
    try:
        loaded_df = pd.read_csv(spot_stats_path)
        print(f"Verification: CSV contains {len(loaded_df)} rows (expected {len(spot_stats)})")
        if len(loaded_df) != len(spot_stats):
            print("WARNING: Row count mismatch!")
    except Exception as e:
        print(f"Error verifying CSV: {e}")   


if __name__ == "__main__":
    data_dir = "/groups/sgro/sgrolab/jennifer/predicty/"
    model_time = "20250903_1715"
    shift_forward = 32
    window_size = 48
    prob_thresh = 0.1
    predict_z = True
    z_model_time = "20251107_1721"  # model to use for predicting z positions if predict_z is True

    main(data_dir, model_time, shift_forward, window_size, prob_thresh, predict_z, z_model_time)
