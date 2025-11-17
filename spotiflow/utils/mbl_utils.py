import dask.array as da
import numpy as np

from skimage.feature import peak_local_max
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
    """"Manually crop a 3D image around an annotation point for prediction."""
    cz, cy, cx = crop_size
    assert len(crop_size) == 3, "crop_size must be a 3-tuple (z, y, x)"
    img_z, img_y, img_x = image.shape[:3]
    # ensure crop fits within the image dimensions
    assert cz <= img_z and cy <= img_y and cx <= img_x, "crop_size must be <= image.shape in all dimensions"

    z, y, x = map(int, annotation)

    # center the crop on the annotation then clamp to image bounds so crop size is exact
    z_start = 0
    y_start = int(y - cy // 2)
    x_start = int(x - cx // 2)

    z_start = 0 # crop full z dimension
    y_start = max(0, min(y_start, img_y - cy))
    x_start = max(0, min(x_start, img_x - cx))

    z_end = z_start + img_z
    y_end = y_start + cy
    x_end = x_start + cx

    cropped_image = image[z_start:z_end, y_start:y_end, x_start:x_end]
    cropped_annotation = np.array([z - z_start, y - y_start, x - x_start])

    return cropped_image, cropped_annotation


def match_previous_window(current_spot, previous_spots):
    # Match current spot to previous spots
    prev_matches = points_matching(
        p1=previous_spots,
        p2=np.array([current_spot]),  # t,y,x
        cutoff_distance=75,
        eps=1e-8,
    )
    if prev_matches.tp > 0:
        assert len(prev_matches.matched_pairs) == 1, "Expected exactly one matched pair"

        prev_spot_idx = prev_matches.matched_pairs[0][0]
        curr_spot_idx = prev_spot_idx

    else:
        curr_spot_idx = None

    return curr_spot_idx


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


def predict_heatmap(img, model, prob_thresh, num_peaks=8):
    """Predict spots using heatmap peaks."""
    _, details = model.predict(
        img,
        subpix=True,
        min_distance=75,
        prob_thresh=prob_thresh,
        #n_tiles=n_tiles, # change if you run out of memory
        device="cuda",
        verbose=False,
    )

    print("Analyzing heatmap peaks...")
    heatmap = details.heatmap
    spots = peak_local_max(
        heatmap,
        min_distance=75,
        threshold_abs=prob_thresh,
        num_peaks=num_peaks,
        exclude_border=False,
    )
    probs = heatmap[spots[:, 0], spots[:, 1], spots[:, 2]]

    return spots, probs, details