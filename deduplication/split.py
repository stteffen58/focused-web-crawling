import csv
import random
import os


path = 'goldstandard-url/'

def train(line, csv_file):
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerow(line)

def test(line, csv_file):
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerow(line)

for filename in os.listdir(path):
    if os.path.isfile(path + filename):
        csv_file = open(path + filename, 'r')
        csv_file_train = open(path + 'train/' + filename, 'w')
        csv_file_test = open(path + 'test/' + filename, 'w')

        reader = csv.reader(csv_file, delimiter=';')
        print (filename)
        for line in reader:
            url1 = line[0]
            url2 = line[1]
            if line[3] == 'true':
                if random.randint(0,1) == 1:
                    train(line, csv_file_train)
                else:
                    test(line, csv_file_test)

        csv_file.close()
        csv_file_test.close()
        csv_file_train.close()
