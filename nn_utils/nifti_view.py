import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def nii_view(image_num: str, path: str):
    # image_num: 
    # path: 
    # Example image_num = 56, path = 'media/ncc/Tasks/Task77_KidneyTumour'
    cur_dir = os.getcwd()
    main_path = os.path.join(cur_dir, path)
    train_image_dir = os.path.join(main_path, 'imagesTr')
    train_label_dir = os.path.join(main_path, 'labelsTr')
    test_image_dir = os.path.join(main_path, 'imagesTs')

    TRAIN_NAME = 'training{}.nii.gz'.format(NUM)
    TEST_NAME = 'test{}.nii.gz'.format(NUM)

