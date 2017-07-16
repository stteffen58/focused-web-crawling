import os
import warc
import gzip
import csv


warc_path = '/home/steffen/PycharmProjects/focused-web-crawling/deduplication/samples/'
gold_path = '/home/steffen/PycharmProjects/focused-web-crawling/deduplication/goldstandard/'

#for file in os.listdir(warc_path):
file = 'walmart_sample.warc.gz'
filename = warc_path + file
print (filename)
with gzip.open(filename, 'rb') as gfz:
    id_uri = {}
    for record in warc.WARCFile(fileobj=gfz):
        id_uri[record['WARC-Record-ID']] = record['WARC-Target-URI']

    domain = filename[filename.rindex(('/')) + 1:filename.rindex(('_'))]
    gold_filename = domain
    csv_file = open('/home/steffen/PycharmProjects/focused-web-crawling/deduplication/goldstandard-url/' + domain + '.csv', 'w')
    writer = csv.writer(csv_file, delimiter=',',quotechar='"',quoting=csv.QUOTE_NONNUMERIC)
    with open(gold_path + gold_filename,'r') as f:
        for line in f:
            line = line.strip()
            splits = line.split(' ')
            row = [id_uri[splits[0]],id_uri[splits[1]],splits[2],splits[3]]
            writer.writerow(row)
    csv_file.close()