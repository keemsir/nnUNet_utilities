# nnUNet_utilities
A Python library for nnUNet

Utilities for nnUNet

pip install git+https://github.com/keemsir/nnUNet_utilities.git

ex) import nn_utils

from nn_utils import path_utils



nnU-Net은 다양한 medical image의 아키텍처와 파라미터의 분석을 토대로

높은 수준의 전처리 프로세스와 학습 플랜을 제공하는 end to end pipe line network이다.

nnU-Net은 공식적으로 input 데이터로써 nifti 파일만을 지원하는데

우리가 주로 다루는 medical image 의 데이터 포멧은 dicom이 대부분이다.

nnU-Net의 pipe line을 사용하기 위해서 dicom file을 nifti 파일로의 변환이 필요하다.

또한 nnU-Net의 파일 경로 구조와 파일명 등을 변경해야 할 필요가 있다.

dicom -> nifti 변환기는 물론, png -> nifti 변환기, json 파일 생성기 등등

nnU-Net을 사용하기 위한 utils로 구성되어있다.

nifti 파일은 nii.gz 혹은 nii 파일 포멧으로 dcm 파일을 하나로 압축해놓은 포멧으로, 보통 3d 이미지와 affine, 단위 정보 등을 포함하고있다.





'''

nnU-Net is an end to end pipeline network that provides a high-level preprocessing process and learning plan based on

the analysis of the architecture and parameters of various medical images.


nnU-Net officially supports only nifti files as input data, and most of the data formats of medical images we deal with are dicom files.

In order to use the pipe line of nnU-Net, it is necessary to convert the dicom file into a nifti file.


It consists of utils for using nnU-Net, such as dicom -> nifti converter, png -> nifti converter, and json file generator.

'''

---

# 1. 환경설정

nnU-Net을 학습시키기 위해서 최소 10GB 의 GPU memory가 필요하다.

초기설정을위해서 


# 2. 데이터 전처리

# 3. 학습

# 4. 예측

# 5. 앙상블

---


