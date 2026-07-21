import pandas as pd
import matplotlib as plt

file = '/home/lorenzo-mobilia/public_html/EasyResNetPaper-v1/S1/results/test_scores.csv'
file_save = '/home/lorenzo-mobilia/public_html/EasyResNetPaper-v1/S1/results/test_scores_added_inj_label.csv'

dataset_df = pd.read_csv(file, sep = ',')
dataset_df['Label'] =dataset_df['Label'].fillna('Injection')
dataset_df.to_csv(file_save, index = False)
