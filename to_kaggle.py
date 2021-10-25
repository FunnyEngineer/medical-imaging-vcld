import csv
import argparse

parser = argparse.ArgumentParser(description='Convert your predicted test.csv to kaggle format.')
parser.add_argument('--pred_csv_path', type=str, default='./pred.csv',
                    help='Path to your predicted csv file.')
parser.add_argument('--out_csv_path', type=str, default='./pred_kaggle.csv',
                    help='Path to your output csv file.')
args = parser.parse_args()


kaggle_eval_classes = ['ich', 'ivh', 'sah', 'sdh', 'edh']
with open(args.pred_csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    output_rows = []
    for row_idx, row in enumerate(csv_reader):
        if row_idx == 0:
            continue
        for cls_idx, cls in enumerate(kaggle_eval_classes):
            ID_single = row[1].split('.')[0] + '_' + cls
            output_row = [ID_single, row[cls_idx + 2]]
            output_rows.append(output_row)

            
with open(args.out_csv_path, mode='w') as csv_file:
    fieldnames = ['ID', 'prediction']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for row in output_rows:
        writer.writerow({'ID': row[0], 'prediction': row[1]})
    
    
