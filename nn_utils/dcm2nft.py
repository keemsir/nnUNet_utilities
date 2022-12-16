import os
import numpy as np
import nibabel as nib
import pydicom


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def dcm2nifti(dcm_folder: str, save_folder: str) -> None:
    # dcm_folder : [train_dir, test_dir] dicom file path
    # save_folder : [imagesTr, imagesTs] Save Folder path
    maybe_mkdir_p(save_folder)
    print('Creating "{}" Image..'.format(os.path.basename(os.path.normpath(save_folder))))
    DCM_files = os.listdir(dcm_folder)
    for DCM_file in DCM_files:
        images = np.empty([512, 512, 0], dtype=np.single)
        DCM_path = os.path.join(dcm_folder, DCM_file)
        DCM_path_sort = sorted(os.listdir(DCM_path))
        for f in DCM_path_sort:
            os.path.join(DCM_path, f)
            f_dcm = pydicom.read_file(os.path.join(DCM_path, f))
            slice = f_dcm.pixel_array
            f_slope = f_dcm.RescaleSlope
            f_intercept = f_dcm.RescaleIntercept
            image = (slice*f_slope + f_intercept)
            image = np.expand_dims(image, axis=2)
            images = np.append(images, image, axis=2)

        niim = nib.Nifti1Image(images, affine=np.eye(4)) # affine = [[1., 0., 0., 0.], [0., 1., 0., 0.],[0., 0., 1., 0.],[0., 0., 0., 1.]]
        nib.save(niim, os.path.join(save_folder, '{}.nii.gz'.format(DCM_file)))
    print('"{}" Image Completed !!'.format(os.path.basename(os.path.normpath(save_folder))))