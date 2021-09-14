import json
import os
import csv


cur_dir = '/home/ncc/PycharmProjects/nnUNet'
cv_dir = os.path.join(cur_dir, 'media/ncc/nnunet_trained_models/nnUNet/3d_fullres/Task577_KidneyTumour/nnUNetTrainerV2__nnUNetPlansv2.1/cv_niftis_postprocessed/')
json_dir = os.path.join(cv_dir, 'summary.json')


with open(json_dir, 'r') as json_file:
    cv_json = json.load(json_file)
cv_json_len = len(cv_json['results']['all'])

#Save path setting
os.chdir(cv_dir)

with open('cv_summary.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["NUM", "Patient_Num", "Accuracy", "Dice", "Jaccard", "Precision", "Recall",
                         "Total Positives Reference", "Total Positives Test", "True Negative Rate"])

    for i in range(cv_json_len):
        cv_col_1 = cv_json['results']['all'][i]['reference'][-18:-7]
        cv_col_2 = cv_json['results']['all'][i]['2']['Accuracy']
        cv_col_3 = cv_json['results']['all'][i]['2']['Dice']
        cv_col_4 = cv_json['results']['all'][i]['2']['Jaccard']
        cv_col_5 = cv_json['results']['all'][i]['2']['Precision']
        cv_col_6 = cv_json['results']['all'][i]['2']['Recall']
        cv_col_7 = cv_json['results']['all'][i]['2']['Total Positives Reference']
        cv_col_8 = cv_json['results']['all'][i]['2']['Total Positives Test']
        cv_col_9 = cv_json['results']['all'][i]['2']['True Negative Rate']

        csv_writer.writerow(
            [i, cv_col_1, cv_col_2, cv_col_3, cv_col_4, cv_col_5, cv_col_6, cv_col_7, cv_col_8, cv_col_9])

#Default Path setting
os.chdir(cur_dir)