import csv
import os

def csv_to_tsv(path_to_csv_file, path_to_tsv_file):
    with open(path_to_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        with open(path_to_tsv_file,'w') as tsv_file:
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    #print(row)
                    SEQUENCE_ID = row[0]
                    SEQUENCE = row[1]
                    CREATURE = row[2][8:]
                    LABEL = row[3][5:]
                    tsv_file.write('{}\t{}\t{}\n'.format(SEQUENCE, CREATURE, LABEL))
                    line_count += 1

def train_test_dev_split(path_to_tsv_file, train=0.7, test=0.2):
    with open(path_to_tsv_file) as tsv_file:
        lines = [line for line in tsv_file]
        line_count = 0
        line_count = len(lines)
        num_train_lines = int(train*line_count)
        num_test_lines = int(test*line_count)
        num_dev_lines = int((1-train-test)*line_count)

        with open (os.path.join(os.getcwd(), 'tsv_files','train.tsv'), 'a') as train_tsv:
            train_tsv.writelines(lines[0:num_train_lines])

        with open (os.path.join(os.getcwd(), 'tsv_files','test.tsv'), 'a') as test_tsv:
            test_tsv.writelines(lines[num_train_lines:num_train_lines+num_test_lines])

        with open (os.path.join(os.getcwd(), 'tsv_files','dev.tsv'), 'a') as dev_tsv:
            dev_tsv.writelines(lines[num_train_lines+num_test_lines:])








#train_test_dev_split(path_to_tsv_file=os.path.join(os.getcwd(),'tsv_files','enzymes.tsv'))

csv_to_tsv(path_to_csv_file=os.path.join(os.getcwd(),'csv_files','Train.csv'), path_to_tsv_file=os.path.join(os.getcwd(),'tsv_files','enzymes.txt'))