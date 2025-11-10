import json
import dask.array as da
import numpy as np
import napari
import torch
import os
import pandas as pd

from spotiflow.model import Spotiflow
from spotiflow.augmentations.transforms3d import Crop3D
from spotiflow.data.spots2dt import Spots2DT
from spotiflow.utils.matching import points_matching

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
    
def crop_manual(image: da.Array, annotation: np.ndarray, crop_size: tuple):
    cz, cy, cx = crop_size
    assert len(crop_size) == 3, "crop_size must be a 3-tuple (z, y, x)"
    img_z, img_y, img_x = image.shape[:3]
    # ensure crop fits within the image dimensions
    assert cz <= img_z and cy <= img_y and cx <= img_x, "crop_size must be <= image.shape in all dimensions"

    z, y, x = map(int, annotation)

    # center the crop on the annotation then clamp to image bounds so crop size is exact
    z_start = int(z - cz // 2)
    y_start = int(y - cy // 2)
    x_start = int(x - cx // 2)

    z_start = max(0, min(z_start, img_z - cz))
    y_start = max(0, min(y_start, img_y - cy))
    x_start = max(0, min(x_start, img_x - cx))

    z_end = z_start + cz
    y_end = y_start + cy
    x_end = x_start + cx

    cropped_image = image[z_start:z_end, y_start:y_end, x_start:x_end]
    cropped_annotation = np.array([z - z_start, y - y_start, x - x_start])

    return cropped_image, cropped_annotation

def extract_matched_pairs(stats, annotation, spots):
    '''Extract matched pairs of annotations and predicted spots based on matching statistics.
    Args:
        stats: Matching statistics object containing indices of matched pairs.
        annotations: Array of annotation coordinates (t, z, y, x).
        spots: Array of predicted spot coordinates (z, y, x).
    Returns:
        Array of matched pairs with shape (N, 2, 4) where N is the number of matched pairs,
        and each pair contains the annotation and corresponding predicted spot coordinates 
        (t, x, y, z).
    '''
    pairs_ids = np.array(stats.matched_pairs)
    print(f"pairs_ids: {pairs_ids}")
    print(f"annotation: {annotation}")
    pairs = np.zeros((pairs_ids.shape[0], 2, 4), dtype=np.float32)
    pairs[:, 0, :] = annotation[0][:, [0, 3, 2, 1]] # reorder to t,x,y,z
    pairs[:, 1, 1:] = spots[pairs_ids[:, 1]][:, [2, 1, 0]] # add x,y,z
    pairs[:, 1, 0] = pairs[:, 0, 0] # add t from annotations
    return pairs

def main(data_dir: str, model_time: str, early_predict_frames: int = 1, prob_thresh: float = 0.5):

    print(f"Predicting with model {model_time}")

    model_path = "/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-" + model_time
    model = Spotiflow.from_folder(model_path, map_location="cuda")
    model.to(torch.device("cuda"))

    with open(data_dir, "r") as f:
        data = json.load(f)
    val_data_list = list(data.items())[-3:]  # last 3 are validation datasets

    # Initialize metrics dataframe
    metrics = pd.DataFrame(columns=["dataset", "TP", "FP", "FN"])
    dist_error_list = []
    z_dist_error_list = []
    matched_pairs_data = []

    for i, zarr_dataset in enumerate(val_data_list):
        print(f"Processing dataset {i}...")
        zarr_dataset_path = zarr_dataset[0]

        # Process annotations
        ds_annotations = zarr_dataset[1]
        # ds_annotations = np.array(ds_annotations) # tzyx

        dataset = load_zarr(os.path.join(zarr_dataset_path, "0", "0"))

        tps = 0
        fps = 0
        fns = 0

        for full_annot in ds_annotations:
            tp = full_annot[0] - early_predict_frames
            img = dataset[tp, 0]  # cells are channel 0
            annot = np.array(full_annot)[-3:]  # z,y,x

            # Crop image and annotation manually
            cropped_img, cropped_annot = crop_manual(img, annot, crop_size=(64, 256, 256))

            # Predict
            spots, details = model.predict(
                img=cropped_img.astype(np.float32).compute(),
                subpix=True,
                min_distance=75,
                prob_thresh=prob_thresh,
                #n_tiles=n_tiles, # change if you run out of memory
                device="cuda",
            )

            print(f"p1: {np.array(spots)}, p2: {np.array([cropped_annot])}")
            # Evaluate
            stats = points_matching(
                        p1=np.array([cropped_annot]),
                        p2=np.array(spots),
                        cutoff_distance=20,
                        eps=1e-8,
                    )
            print(f"TP: {stats.tp}, FP: {stats.fp}, FN: {stats.fn}")
            tps += stats.tp
            fps += stats.fp
            fns += stats.fn

            viewer = napari.Viewer()
            # viewer.add_image(imgs[-1])
            #viewer.add_points(all_annotations[pred_dataset], size=20, name="pts", symbol="disc", border_color="red", face_color="red")
            viewer.add_image(cropped_img, name="img")
            viewer.add_points(spots, size=20, name="pts", symbol="disc", border_color="blue", face_color="blue")
            viewer.add_points(cropped_annot, size=20, name="annotations", symbol="disc", border_color="red", face_color="red")
            viewer.add_image(details.heatmap, name="pred_heatmap", colormap="magma", blending="additive", opacity=0.6)
            viewer.add_image((details.flow+1)*0.5, name="flow")
            napari.run()

            if len(stats.matched_pairs) > 0:
                assert len(stats.matched_pairs) == 1, "Expected exactly one matched pair"
                spot = spots[stats.matched_pairs[0][0]]
                dist_error_list.extend(stats.dist.tolist()) # calculated as error in t, x, y where t has been shifted forward
                z_dist_error = abs(cropped_annot[0] - spot[0])
                z_dist_error_list.append(z_dist_error)
                cropped_annot = [int(tp), int(cropped_annot[0]), int(cropped_annot[1]), int(cropped_annot[2])]  # add time coordinate 
                matched_pairs_data.append({
                    "dataset": zarr_dataset_path,
                    "matched_pair": [full_annot, cropped_annot, spot.tolist()],
                    "distance_error": stats.dist.tolist(),
                    "z_distance_error": float(z_dist_error),
                })
        
        print(f"Dataset {i} - TP: {tps}, FP: {fps}, FN: {fns}")
        metrics = pd.concat([metrics, pd.DataFrame({"dataset": [zarr_dataset_path], "TP": [tps], "FP": [fps], "FN": [fns]})], ignore_index=True)

    # Save metrics to CSV
    avg_dist_error = np.mean(dist_error_list) if len(dist_error_list) > 0 else 0
    print(f"Average distance error across all datasets: {avg_dist_error}")
    avg_z_dist_error = np.mean(z_dist_error_list) if len(z_dist_error_list) > 0 else 0
    print(f"Average z distance error across all datasets: {avg_z_dist_error}")
    # Add average distance error to metrics dataframe
    metrics["avg_distance_error"] = avg_dist_error
    metrics["avg_z_distance_error"] = avg_z_dist_error
    metrics_path = os.path.join(model_path, f"prediction_metrics_{prob_thresh}pt.csv")
    metrics.to_csv(metrics_path, index=False)

    # Save matched pairs data to JSON
    matched_pairs_path = os.path.join(model_path, f"matched_pairs_{prob_thresh}pt.json")
    with open(matched_pairs_path, "w") as f:
        json.dump(matched_pairs_data, f, indent=4)

if __name__ == "__main__":
    data_dir = "/groups/sgro/sgrolab/jennifer/predicty/training_data_server.json"
    main(data_dir, 
         model_time = "20251107_1721",
         early_predict_frames=1,
         prob_thresh = 0.2,
         )