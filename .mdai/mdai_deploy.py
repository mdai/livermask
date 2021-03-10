import os
from io import BytesIO
import numpy as np
import pydicom
import tempfile
import h5py

import nibabel as nib
import dicom2nifti
from scipy.ndimage import zoom
from tensorflow.python.keras.models import load_model

import warnings
from skimage.morphology import remove_small_holes, binary_dilation, binary_erosion, ball
from skimage.measure import label, regionprops
from helper import intensity_normalization, sort_dicoms
import dicom2nifti.settings as settings

settings.disable_validate_orthogonal()
settings.disable_validate_slice_increment()

warnings.filterwarnings("ignore", ".*output shape of zoom.*")


class MDAIModel:
    def __init__(self):
        self.modelpath = os.path.join(os.path.dirname(__file__), "../model.h5")
        self.tempdir = tempfile.mkdtemp()

    def predict(self, data):
        """
        See https://github.com/mdai/model-deploy/blob/master/mdai/server.py for details on the
        schema of `data` and the required schema of the outputs returned by this function.
        """
        input_files = data["files"]
        input_annotations = data["annotations"]
        input_args = data["args"]

        outputs, dicom_files = [], []
        for file in input_files:
            if file["content_type"] != "application/dicom":
                continue

            dicom_files.append(pydicom.dcmread(BytesIO(file["content"])))

        dicom_files = sort_dicoms(dicom_files)

        dicom2nifti.convert_dicom.dicom_array_to_nifti(
            dicom_files,
            output_file=os.path.join(
                self.tempdir, dicom_files[0].SeriesInstanceUID + ".nii.gz"
            ),
            reorient_nifti=True,
        )

        nib_volume = nib.load(
            os.path.join(self.tempdir, dicom_files[0].SeriesInstanceUID + ".nii.gz")
        )

        # new_spacing = [1.0, 1.0, 1.0]
        # resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
        # data = resampled_volume.get_data().astype("float32")

        data = nib_volume.get_fdata().astype("float32")
        curr_shape = data.shape

        # resize to get (512, 512) output images
        img_size = 512
        data = zoom(
            data, [img_size / data.shape[0], img_size / data.shape[1], 1.0], order=1
        )

        # intensity normalization
        intensity_clipping_range = [
            -150,
            250,
        ]  # HU clipping limits (Pravdaray's configs)
        data = intensity_normalization(
            volume=data, intensity_clipping_range=intensity_clipping_range
        )

        # fix orientation
        data = np.rot90(data, k=1, axes=(0, 1))
        data = np.flip(data, axis=0)

        model = load_model(self.modelpath, compile=False)
        print("predicting...", flush=True)
        # predict on data
        pred = np.zeros_like(data).astype(np.float32)
        for i in range(data.shape[-1]):
            pred[..., i] = model.predict(
                np.expand_dims(
                    np.expand_dims(np.expand_dims(data[..., i], axis=0), axis=-1),
                    axis=0,
                )
            )[0, ..., 1]
        del data

        # threshold
        pred = (pred >= 0.4).astype(int)

        # fix orientation back
        pred = np.flip(pred, axis=0)
        pred = np.rot90(pred, k=-1, axes=(0, 1))

        print("resize back...", flush=True)
        # resize back from 512x512
        pred = zoom(
            pred, [curr_shape[0] / img_size, curr_shape[1] / img_size, 1.0], order=1
        )
        pred = (pred >= 0.5).astype(np.float32)

        print("morphological post-processing...", flush=True)
        # morpological post-processing
        # 1) first erode
        pred = binary_erosion(pred.astype(bool), ball(3)).astype(np.float32)

        # 2) keep only largest connected component
        labels = label(pred)
        regions = regionprops(labels)
        if regions:
            area_sizes = []
            for region in regions:
                area_sizes.append([region.label, region.area])
            area_sizes = np.array(area_sizes)
            tmp = np.zeros_like(pred)
            tmp[labels == area_sizes[np.argmax(area_sizes[:, 1]), 0]] = 1
            pred = tmp.copy()
            del tmp, labels, regions, area_sizes
        else:
            for i in range(len(dicom_files)):
                outputs.append(
                    {
                        "type": "NONE",
                        "study_uid": str(dicom_files[i].StudyInstanceUID),
                        "series_uid": str(dicom_files[i].SeriesInstanceUID),
                        "instance_uid": str(dicom_files[i].SOPInstanceUID),
                    }
                )
            return outputs

        # 3) dilate
        pred = binary_dilation(pred.astype(bool), ball(3))

        # 4) remove small holes
        pred = remove_small_holes(
            pred.astype(bool), area_threshold=0.001 * np.prod(pred.shape)
        ).astype(np.float32)

        print("saving...", flush=True)
        pred = pred.astype(np.uint8)

        if len(pred.shape) != 3:
            pred = np.expand_dims(pred, -1)

        for i, mask in enumerate(np.transpose(np.rot90(pred), (2, 0, 1))):
            if np.sum(mask) != 0:
                outputs.append(
                    {
                        "type": "ANNOTATION",
                        "study_uid": str(dicom_files[i].StudyInstanceUID),
                        "series_uid": str(dicom_files[i].SeriesInstanceUID),
                        "instance_uid": str(dicom_files[i].SOPInstanceUID),
                        "class_index": 0,
                        "data": {"mask": mask.tolist()},
                    }
                )
            else:
                outputs.append(
                    {
                        "type": "NONE",
                        "study_uid": str(dicom_files[i].StudyInstanceUID),
                        "series_uid": str(dicom_files[i].SeriesInstanceUID),
                        "instance_uid": str(dicom_files[i].SOPInstanceUID),
                    }
                )

        os.remove(
            os.path.join(self.tempdir, dicom_files[0].SeriesInstanceUID + ".nii.gz")
        )
        return outputs
