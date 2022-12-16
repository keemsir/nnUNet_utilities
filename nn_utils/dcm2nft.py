import os
import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk

from nn_utils.path_utils import maybe_mkdir_p



# dcm2nifti
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
    print('"{}" converted from dicom to Nifti !!'.format(os.path.basename(os.path.normpath(save_folder))))


# png2nifti
def png2nifti(png_folder: str, save_folder: str):
    # png_folder : png file folder
    # save_folder : Save folder path
    PNG_files = os.listdir(png_folder)
    print('Creating "{}" Label..'.format(os.path.basename(os.path.normpath(save_folder))))
    maybe_mkdir_p(save_folder)
    for PNG_file in PNG_files:
        labels = np.empty([512, 512, 0], dtype=np.uint8)
        PNG_path = os.path.join(png_folder, PNG_file)
        PNG_path_sort = sorted(os.listdir(PNG_path))
        for f in PNG_path_sort:
            os.path.join(PNG_path, f)
            f_png = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(PNG_path, f)))

            label = np.expand_dims(f_png, axis=2)
            labels = np.append(labels, label, axis=2)

        nila = nib.Nifti1Image(labels, affine=np.eye(4))
        nib.save(nila, os.path.join(save_folder, '{}.nii.gz'.format(PNG_file)))
    print('"{}" converted from png to Nifti !!'.format(os.path.basename(os.path.normpath(save_folder))))

