import numpy as np


def sort_dicoms(dicoms):
    """
    Sort the dicoms based om the image possition patient

    :param dicoms: list of dicoms
    """
    # find most significant axis to use during sorting
    # the original way of sorting (first x than y than z) does not work in certain border situations
    # where for exampe the X will only slightly change causing the values to remain equal on multiple slices
    # messing up the sorting completely)
    dicom_input_sorted_x = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[0]))
    dicom_input_sorted_y = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[1]))
    dicom_input_sorted_z = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[2]))
    diff_x = abs(
        dicom_input_sorted_x[-1].ImagePositionPatient[0]
        - dicom_input_sorted_x[0].ImagePositionPatient[0]
    )
    diff_y = abs(
        dicom_input_sorted_y[-1].ImagePositionPatient[1]
        - dicom_input_sorted_y[0].ImagePositionPatient[1]
    )
    diff_z = abs(
        dicom_input_sorted_z[-1].ImagePositionPatient[2]
        - dicom_input_sorted_z[0].ImagePositionPatient[2]
    )
    if diff_x >= diff_y and diff_x >= diff_z:
        return dicom_input_sorted_x
    if diff_y >= diff_x and diff_y >= diff_z:
        return dicom_input_sorted_y
    if diff_z >= diff_x and diff_z >= diff_y:
        return dicom_input_sorted_z


def intensity_normalization(volume, intensity_clipping_range):
    result = np.copy(volume)

    result[volume < intensity_clipping_range[0]] = intensity_clipping_range[0]
    result[volume > intensity_clipping_range[1]] = intensity_clipping_range[1]

    min_val = np.amin(result)
    max_val = np.amax(result)
    if (max_val - min_val) != 0:
        result = (result - min_val) / (max_val - min_val)

    return result
