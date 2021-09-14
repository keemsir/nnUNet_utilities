import numpy as np
import os
import nibabel as nib
from PIL import Image
from typing import List
import tqdm


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def nifti_files(folder: str, join: bool = True, sort: bool = True) -> List[str]:
    return subfiles(folder, join=join, sort=sort, suffix='.nii.gz')


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def make_submission(input_dir, save_path, num_classes):
    f = open(save_path, "w")
    f.write("Id,EncodedPixels\n")
    case_names = sorted(os.listdir(input_dir))
    for case_name in case_names:
        case_dir = os.path.join(input_dir, case_name)
        slice_names = sorted(os.listdir(case_dir))
        for class_idx in range(1, num_classes):
            mask_stack = np.array([], dtype='uint8')
            for slice_name in slice_names:
                slice_img_path = os.path.join(case_dir, slice_name)
                slice_img = np.array(Image.open(slice_img_path).convert('L')).flatten()
                slice_mask = np.equal(slice_img, class_idx) * 255
                mask_stack = np.hstack([mask_stack, slice_mask])
            data_id = f'{case_name}_{class_idx}'
            enc = rle_to_string(rle_encode(mask_stack))
            line = f'{data_id},{enc}'
            f.write(line + "\n")
    f.close()

# Path Setting
cur_dir = os.getcwd()
task_dir = os.path.join(cur_dir, 'OUTPUT_DIRECTORY/581/2d/')
maybe_mkdir_p(task_dir)
task_files = nifti_files(task_dir)

# png create
save_folder_name = os.path.basename(os.path.normpath(task_dir))
save_folder = os.path.join(cur_dir, 'png/{}/'.format(save_folder_name))
maybe_mkdir_p(save_folder)

for task_file in task_files:
    array = np.array(nib.load(task_file).dataobj)
    _, _, array_len = array.shape
    task_name = os.path.basename(task_file)[0:7]
    task_folder = os.path.join(save_folder, task_name)
    maybe_mkdir_p(task_folder)

    for i in range(array_len):
        array_i = array[:, :, i]
        im = Image.fromarray(array_i)
        im.save(task_folder+"/{0:05d}.png".format(i), format="png")

cur_dir = '/home/ncc/PycharmProjects/nnUNet'
os.chdir(task_dir)

print('csv file Saving..')
make_submission(save_folder, 'submission.csv', 3)
print('Save Complete ! csv file save path : {}'.format(task_dir))
os.chdir(cur_dir)


## PNG -> submission.csv
import numpy as np
import os
import nibabel as nib
from PIL import Image
from typing import List
import tqdm


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def nifti_files(folder: str, join: bool = True, sort: bool = True) -> List[str]:
    return subfiles(folder, join=join, sort=sort, suffix='.nii.gz')


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def make_submission(input_dir, save_path, num_classes):
    f = open(save_path, "w")
    f.write("Id,EncodedPixels\n")
    case_names = sorted(os.listdir(input_dir))
    for case_name in case_names:
        case_dir = os.path.join(input_dir, case_name)
        slice_names = sorted(os.listdir(case_dir))
        for class_idx in range(1, num_classes):
            mask_stack = np.array([], dtype='uint8')
            for slice_name in slice_names:
                slice_img_path = os.path.join(case_dir, slice_name)
                slice_img = np.array(Image.open(slice_img_path).convert('L')).flatten()
                slice_mask = np.equal(slice_img, class_idx) * 255
                mask_stack = np.hstack([mask_stack, slice_mask])
            data_id = f'{case_name}_{class_idx}'
            enc = rle_to_string(rle_encode(mask_stack))
            line = f'{data_id},{enc}'
            f.write(line + "\n")
    f.close()



# Path Setting
cur_dir = '/home/ncc/PycharmProjects/nnUNet'
task_dir = '/home/ncc/PycharmProjects/nnUNet/png'

save_folder = os.path.join(cur_dir, 'png/staple_m4_0501/')
# maybe_mkdir_p(save_folder)

os.chdir(task_dir)

print('csv file Saving..')
make_submission(save_folder, 'submission.csv', 3)
print('Save Complete ! csv file save path : {}'.format(save_folder))
os.chdir(cur_dir)
