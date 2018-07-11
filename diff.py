#!/bin/python3
import sys
import os
import getopt
import csv
from collections import defaultdict
import tlsh

from multiprocessing.dummy import Pool

def usage():
    print("python3 ./diff.py [file directory] [metadata file]")

def lsh(filename):
    if os.path.getsize(filename) < 256:
        raise ValueError("{} must be at least 256 bytes".format(filename))

    print(filename)
    return tlsh.hash(open(filename, 'rb').read())

def diff_hash(one, two):
    return tlsh.diff(one, two)

def list_files(directory):
    f = []
    for (dirpath, _, filenames) in os.walk(directory):
        for name in filenames:
            f.append(os.path.join(dirpath, name))
    return f

def parse_metadata(filename):
    contents = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            #Remove the md5 and sha1 hashes since they're useless to me
            contents.append(row[:-2])
    return contents

try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:m:", ["help", "directory", "metadata"])
except getopt.GetoptError as err:
    print(err) # will print something like "option -a not recognized"
    usage()
    exit(2)

directory = ""
meta = ""

for o, a in opts:
    if o in ("-d", "--directory"):
        directory = a
    elif o in ("-h", "--help"):
        usage()
        exit()
    elif o in ("-m", "--metadata"):
        meta = a

if not directory or not meta:
    print("Program must be provided a file directory path and a metadata file")
    exit(1)

file_list = list_files(directory)
hash_list = []

with Pool(8) as p:
    hash_list = p.map(lsh, file_list)
    for h in sorted(hash_list):
        print(h)

#count = 0
#for i in sorted(test_col)[:n]:
    #print("\nDistance " + str(i))
    #for elem in test_col[i]:
        #print(elem[0])
        #print(elem[1])
        #count += 1
        #if count >= n:
            #exit()
