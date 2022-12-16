import os
import json
from collections import OrderedDict


## Json creating
def mk_json(save_dir: str):

    imagesTr = os.path.join(save_dir, 'imagesTr')
    imagesTs = os.path.join(save_dir, 'imagesTs')

    overwrite_json_file = True
    json_file_exist = False

    if os.path.exists(os.path.join(save_dir, 'dataset.json')):
        print('dataset.json already exist!')
        json_file_exist = True

    if json_file_exist == False or overwrite_json_file:

        json_dict = OrderedDict()
        json_dict['name'] = "BTCV"
        json_dict['description'] = "Multi-Atlas Labeling Beyond the Cranial Vault"
        json_dict['tensorImageSize'] = "3D"
        json_dict['reference'] = "https://www.synapse.org/#!Synapse:syn3193805/wiki/217752"
        json_dict['licence'] = "CC-BY-NC-SA"
        json_dict['release'] = "22/11/2022"

        json_dict['modality'] = {
            "0": "CT"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "Spleen",
            "2": "R Kidney",
            "3": "L Kidney",
            "4": "Gallbladder",
            "5": "Esophagus",
            "6": "Liver",
            "7": "Stomach",
            "8": "Aorta",
            "9": "IVC",
            "10": "Portal and Splenic Vein",
            "11": "Pancreas",
            "12": "R adrenal gland",
            "13": "L adrenal gland"
        }

        train_ids = sorted(os.listdir(imagesTr))
        test_ids = sorted(os.listdir(imagesTs))
        json_dict['numTraining'] = len(train_ids)
        json_dict['numTest'] = len(test_ids)

        json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in train_ids]

        json_dict['test'] = ["./imagesTs/%s" % i for i in test_ids] #(i[:i.find("_0000")])

        with open(os.path.join(save_dir, "dataset.json"), 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=False)

        if os.path.exists(os.path.join(save_dir, 'dataset.json')):
            if json_file_exist == False:
                print('dataset.json created!')
            else:
                print('dataset.json overwritten!')

    print('Save Path : ' + save_dir)