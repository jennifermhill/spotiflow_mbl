import os
import json
import dask.array as da
import numpy as np
import napari
import torch
import pandas as pd
import tifffile as tiff

from spotiflow.model import Spotiflow
from spotiflow.utils.matching import points_matching

def main(training_data_dir: str, model_time: str, pred_datasets: list[int], shift_forward: int):
    with open(training_data_dir, "r") as f:
        training_data = json.load(f)

    print(f"Predicting with model {model_time}")

    # Process training data
    imgs = []
    annotations = []
    for i, zarr_dataset in enumerate(training_data):
        print(f"Processing dataset {i}...")
        zarr_path = os.path.join(zarr_dataset, "../analysis_mbl/max_projections/maxz")
        # zarr_path = os.path.join(zarr_dataset, "analysis/max_projections/maxz")

        # Process zarr
        try:
            img = da.from_zarr(zarr_path)
        except Exception as e:
            print(f"Skipping dataset {i}: {e}")
            continue
        # img = img[:,0,0,:,:]
        img = img[:,[0,3],:,:].transpose(0,2,3,1)
        
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
    model_path = "/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-" + model_time
    model = Spotiflow.from_folder(model_path, map_location="cuda")
    model.to(torch.device("cuda"))

    tps = 0
    fps = 0
    fns = 0
    recalls = []
    # for ds, cutoff, prob_thresh in zip(pred_datasets, [50, 10, 90], [0.35, 0.3, 0.5]):
    for ds, cutoff, prob_thresh in zip(pred_datasets, [50], [0.35]):
        print("Loading validation image...")
        val_imgs = imgs[ds].astype(np.float32).compute()
        val_imgs_crop = val_imgs[:cutoff]

        print(f"Image shape is: {val_imgs.shape}")


        #n_tiles = tuple(max(s//g, 1) for s, g in zip(val_imgs.shape, (16, 256, 256)))
        #print(n_tiles)
        print("Predicting volume...")

        spots, details = model.predict(
            val_imgs_crop,
            subpix=True,
            min_distance=75,
            prob_thresh=prob_thresh,
            #n_tiles=n_tiles, # change if you run out of memory
            device="cuda",
        )

        viewer = napari.Viewer()
        val_imgs = val_imgs.transpose(3, 0, 1, 2)  # to CTYX for napari
        # viewer.add_image(imgs[-1])
        viewer.add_points(annotations[ds], size=20, name="pts", symbol="disc", border_color="red", face_color="red")
        viewer.add_image(val_imgs[0], name="img_cells")
        viewer.add_image(val_imgs[1], name="img_zramp")
        viewer.add_points(spots, size=20, name="pts", symbol="disc", border_color="blue", face_color="blue")
        viewer.add_image(details.heatmap, name="heatmap")
        viewer.add_image((details.flow+1)*0.5, name="flow")
        napari.run()

        prediction_annotations = annotations[ds]
        # subtract shift forward from time in prediction annotations
        prediction_annotations[:,0] -= shift_forward

        stats = points_matching(
            prediction_annotations,
            spots,
            cutoff_distance=200,
            eps=1e-8,
            )
        print(stats)

        if ds == 16 or ds == 14:
            # create a pandas df with columns for spot, heatmap, and matched
            df = pd.DataFrame(columns=["spot", "heatmap", "prob", "matched"])
            matched_pairs = stats.matched_pairs
            tp_spots = [pair[1] for pair in matched_pairs]
            for i, spot in enumerate(spots):
                # get heatmap for spot
                heatmap = details.heatmap[round(spot[0]), round(spot[1]), round(spot[2])]
                prob = details.prob[i]
                matched = i in tp_spots
                # create df entry
                df_entry = pd.DataFrame({
                    "spot": [i],
                    "heatmap": [heatmap],
                    "prob": [prob],
                    "matched": [matched],
                })
                df = pd.concat([df, df_entry], ignore_index=True)
            print(df)
            # save df to csv
            df.to_csv(f"/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-{model_time}/ds{ds}_stats.csv", index=False)

            #save prediction images as tiff
            # pad_width = ((0, val_imgs.shape[0] - details.heatmap.shape[0]), (0, 0), (0, 0))
            # heatmap_padded = np.pad(details.heatmap, pad_width, mode='constant')
            # img_concat = np.concatenate((val_imgs, heatmap_padded), axis=0)
            # tiff.imwrite(f"/groups/sgro/sgrolab/jennifer/spotiflow_mbl/scripts/outputs/spotiflow-{model_time}/ds{ds}_img.tiff", 
            #               img_concat,
            #               metadata={'axes': 'ctyx'}
            #               )

        tps += stats.tp
        fps += stats.fp
        fns += stats.fn
        recalls.append(stats.recall)

    print(f"Total TPs: {tps}, Total FPs: {fps}, Total FNs: {fns}")


if __name__ == "__main__":
    training_data_dir = "/groups/sgro/sgrolab/jennifer/predicty/training_data_server.json"
    main(training_data_dir,
         model_time = "20250915_1000", 
         pred_datasets=[14,],
         shift_forward=32)