from .spots3d import Spots3DDataset
from typing import Callable, Dict, Literal, Optional, Sequence, Union
import numpy as np
from typing_extensions import Self
import torch
from .. import utils


class Spots2DT(Spots3DDataset):
    def __init__(
        self,
        images: Sequence[np.ndarray],
        centers: Sequence[np.ndarray],
        augmenter: Optional[Callable] = None,
        downsample_factors: Sequence[int] = (1,),
        sigma: float = 1.,
        mode: str = "max",
        compute_flow: bool = False,
        image_files: Optional[Sequence[str]] = None,
        normalizer: Union[Literal["auto"], Callable, None] = "auto",
        defer_normalization: bool = False,
        add_class_label: bool = True,
        grid: Optional[Sequence[int]] = None,
        cropper: Optional[Callable] = None,
    ) -> Self:
        """ Constructor

        Args:
            images (Sequence[np.ndarray]): Sequence of images.
            centers (Sequence[np.ndarray]): Sequence of center coordinates.
            augmenter (Optional[Callable], optional): Augmenter function. If given, function arguments should two (first image, second spots). Defaults to None.
            downsample_factors (Sequence[int], optional): Downsample factors. Defaults to (1,).
            sigma (float, optional): Sigma of Gaussian kernel to generate heatmap. Defaults to 1.
            mode (str, optional): Mode of heatmap generation. Defaults to "max".
            compute_flow (bool, optional): Whether to compute flow from centers. Defaults to False.
            image_files (Optional[Sequence[str]], optional): Sequence of image filenames. If the dataset was not constructed from a folder, this will be None. Defaults to None.
            normalizer (Union[Literal["auto"], Callable, None], optional): Normalizer function. Defaults to "auto" (percentile-based normalization with p_min=1 and p_max=99.8).
            defer_normalization (bool, optional): Whether to defer normalization to data[i] (if normalizer is not None) to save memory Defaults to False.
        """
        super().__init__(
            images=images,
            centers=centers,
            augmenter=augmenter,
            downsample_factors=downsample_factors,
            sigma=sigma,
            mode=mode,
            compute_flow=compute_flow,
            image_files=image_files,
            normalizer=normalizer,
            defer_normalization=defer_normalization,
            add_class_label=add_class_label,
            grid=grid,
        )

        self.cropper = cropper

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img, centers = self.images[index], self._centers[index]

        img, centers = self.cropper(img, centers)
        centers = centers.numpy()

        if self._defer_normalization:
            if img.ndim == 3 or img.shape[0] == 1:
                img = self._normalizer(img.compute())
            elif img.shape[0] == 3:
                img = img.compute()
                img[0] = self._normalizer(img[0])
                img[1] = img[1] / (img[1].max()+1e-8)
                img[2] = img[2] / (img[2].max()+1e-8)

        # Add B (batch) dimension
        img = torch.from_numpy(img.copy()).unsqueeze(0)
        centers = torch.from_numpy(centers.copy()).unsqueeze(0)

        if img.ndim == 4:  # i.e. BDHW, then add C (channel) dimension to BDHW format
            img = img.unsqueeze(1)
        # elif img.ndim == 5:  # i.e. BDHWC, then turn BDWHC format into BCDHW
        #     img = torch.moveaxis(img, -1, 1)
        elif img.ndim != 5:
            raise ValueError(
                f"Image tensor must be 4D (BDHW) or 5D (BCDHW) for {self.__class__.__name__}, got {img.ndim}D."
            )

        img, centers = self.augmenter(img, centers)  # augmenter expects BCDHW format
        img, centers = img.squeeze(0), centers.squeeze(0)  # Remove B (batch) dimension
        if self._compute_flow:
            flow = utils.points_to_flow3d(
                centers.numpy(),
                img.shape[-3:],
                sigma=self._sigma,
                grid=self._grid,
            ).transpose((3, 0, 1, 2))
            flow = torch.from_numpy(flow).float()

        heatmap_lv0 = utils.points_to_prob3d(
            centers.numpy(),
            img.shape[-3:],
            mode=self._mode,
            sigma=self._sigma,
            grid=self._grid,
        )

        # Build target at different resolution levels
        heatmaps = [
            utils.multiscale_decimate(heatmap_lv0, ds, is_3d=True)
            for ds in self._downsample_factors
        ]

        # Cast to tensor and add channel dimension
        ret_obj = {"img": img.float(), "pts": centers.float()}

        if self._compute_flow:
            ret_obj.update({"flow": flow})

        ret_obj.update(
            {
                f"heatmap_lv{lv}": torch.from_numpy(heatmap.copy()).unsqueeze(0)
                for lv, heatmap in enumerate(heatmaps)
            }
        )
        return ret_obj