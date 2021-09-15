import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib



# dataloader

# path
cur_dir = os.getcwd()
main_path = os.path.join(cur_dir, 'tests/KiTS/nifti')

train_image_dir = os.path.join(main_path, 'imagesTr')
train_label_dir = os.path.join(main_path, 'labelsTr')
test_image_dir = os.path.join(main_path, 'imagesTs')

NUM = '050'

TRAIN_NAME = 'training{}.nii.gz'.format(NUM)
TEST_NAME = 'test{}.nii.gz'.format(NUM)

pred1 = os.path.join(train_image_dir, TRAIN_NAME)
pred2 = os.path.join(train_label_dir, TRAIN_NAME)

RESULT_LEN_RAN = np.random.randint(0, 63)
# RESULT_LEN_RAN = 80 # custom number
# RESULT_LEN_RAN = 15
# custom number


pr_label1 = np.array(nib.load(pred1).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
pr_label2 = np.array(nib.load(pred2).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

# pr_augment = augment_gamma(pr_label1)

# pr_augment = rotate_coords_2d(2, pr_label1)

max_rows = 2
max_cols = 5

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Image' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[0, idx].imshow(pr_label1[:, :, idx])
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('Label' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[1, idx].imshow(pr_label2[:, :, idx])


plt.suptitle('Train image NUM : training{}.nii.gz'.format(NUM))
plt.subplots_adjust(wspace=.1, hspace=.2)
plt.show()


## Augmentation visualization
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import imgaug.augmenters as iaa
import random


# dataloader

# path
cur_dir = os.getcwd()
main_path = os.path.join(cur_dir, 'tests/KiTS/nifti')

train_image_dir = os.path.join(main_path, 'imagesTr')
train_label_dir = os.path.join(main_path, 'labelsTr')
test_image_dir = os.path.join(main_path, 'imagesTs')


TRAIN_NAME = 'training{}.nii.gz'.format(NUM)
TEST_NAME = 'test{}.nii.gz'.format(NUM)

image1 = os.path.join(train_image_dir, TRAIN_NAME)
image2 = os.path.join(train_label_dir, TRAIN_NAME)



# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

image_ = np.array(nib.load(image1).dataobj)
label_ = np.array(nib.load(image2).dataobj)

def rot_png(NUM: int, iter: int):
    # NUM : patient number, iter : total iter count
    save_path = 'augmentation/{}'.format(NUM)
    maybe_mkdir_p(save_path)

    for i in range(iter):
        RESULT_LEN_RAN = np.random.randint(0, 63)
        image_ela1 = image_[...,RESULT_LEN_RAN]
        image_ela2 = label_[...,RESULT_LEN_RAN]

        random_NUM = random.randrange(0, 4)
        rotate = iaa.Rot90((random_NUM))

        # random_ROT = random.randrange(-40, 40)
        # rotate = iaa.Affine(rotate=(random_ROT))
        rotated_image = rotate.augment_image(image_ela1)
        rotated_image2 = rotate.augment_image(image_ela2)

        im_merge = np.concatenate((rotated_image[...,None], rotated_image2[...,None]), axis=2)

        # im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.09)

        im_t = im_merge_t[...,0]
        im_mask_t = im_merge_t[...,1]

        max_rows = 2
        max_cols = 2
        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(8, 8))
        for idx in range(max_cols):
            axes[0, 0].axis("off")
            axes[0, 0].imshow(image_ela1)
            axes[0, 1].axis("off")
            axes[0, 1].imshow(im_t)
        plt.ioff()
        for idx in range(max_cols):
            axes[1, 0].axis("off")
            axes[1, 0].imshow(image_ela2)
            axes[1, 1].axis("off")
            axes[1, 1].imshow(im_mask_t)
        plt.ioff()
        plt.savefig(os.path.join(save_path, '{}.png'.format(i)))



## rotate
import imgaug.augmenters as iaa
import random

RESULT_LEN_RAN = np.random.randint(0, 63)
image_ela1 = image_[..., RESULT_LEN_RAN]
image_ela2 = label_[..., RESULT_LEN_RAN]

random_NUM = random.randrange(0,4)
rotate = iaa.Rot90((random_NUM))

# random_ROT = random.randrange(-40, 40)
# rotate = iaa.Affine(rotate=(random_ROT))

rotated_image = rotate.augment_image(image_ela1)
rotated_image2 = rotate.augment_image(image_ela2)

im_merge = np.concatenate((rotated_image[..., None], rotated_image2[..., None]), axis=2)

# im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 3, im_merge.shape[1] * 0.07, im_merge.shape[1] * 0.09)

im_t = im_merge_t[..., 0]
im_mask_t = im_merge_t[..., 1]

max_rows = 2
max_cols = 2
fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(8, 8))
for idx in range(max_cols):
    axes[0, 0].axis("off")
    axes[0, 0].imshow(image_ela1)
    axes[0, 1].axis("off")
    axes[0, 1].imshow(im_t)
for idx in range(max_cols):
    axes[1, 0].axis("off")
    axes[1, 0].imshow(image_ela2)
    axes[1, 1].axis("off")
    axes[1, 1].imshow(im_mask_t)

