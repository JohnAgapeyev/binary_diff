#!/bin/python3
import sys
import os
import getopt
import csv
import itertools
import zipfile
import tarfile
from heapq import nsmallest
from collections import defaultdict
import tlsh
import numpy

from multiprocessing.dummy import Pool

def usage():
    print("python3 ./diff.py [file directory] [metadata file]")

def lsh(data):
    filename = data[0]
    if not data[1] or data[1] == None:
        meta = []
    else:
        meta = [d for d in data[1] if d[-1] == os.path.basename(filename)][0]

    if os.path.getsize(filename) < 256:
        raise ValueError("{} must be at least 256 bytes".format(filename))

    print(filename)

    if tarfile.is_tarfile(filename):
        tar = tarfile.open(filename, 'r')
        for member in tar.getmembers():
            if not member or member.size < 256:
                continue
            try:
                meta.append(tlsh.hash(tar.extractfile(member).read()))
            except:
                continue
    elif zipfile.is_zipfile(filename):
        try:
            z = zipfile.ZipFile(filename)
            for member in z.infolist():
                if not member or member.file_size < 256:
                    continue
                try:
                    meta.append(tlsh.hash(z.read(member)))
                except:
                    continue
        except:
            pass

    file_hash = tlsh.hash(open(filename, 'rb').read())

    if not meta:
        return file_hash
    else:
        return tlsh.hash(str.encode(file_hash + ''.join(meta)))

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
    return contents[1:]

def get_n_closest(n, filenames, adjacency):
    closest = {}
    for f in filenames:
        elem = adj[filenames.index(f)]
        smallest_dists = nsmallest(n + 1, elem)
        smallest_files = []
        old_dist = 0
        for d in smallest_dists:
            #Ignore the file listing itself
            if d == 0:
                continue
            elif d == old_dist:
                continue
            old_dist = d
            if smallest_dists.count(d) > 1:
                prev = 0
                for i in range(smallest_dists.count(d)):
                    dist_filename = smallest_dists.index(d, prev)
                    smallest_files.append((d, filenames[dist_filename]))
                    prev = dist_filename + 1
                continue;
            #Filename indices are analagous to adjacency indices
            smallest_files.append((d, filenames[smallest_dists.index(d)]))
        closest[f] = smallest_files
    return closest

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

if not directory:
    print("Program must be provided a file directory path")
    exit(1)

file_list = list_files(directory)
hash_list = []

if meta:
    meta_contents = parse_metadata(meta)
else:
    meta_contents = None

with Pool() as p:
    hash_list = p.map(lsh, zip(file_list, itertools.repeat(meta_contents)))

    adj = numpy.zeros((len(hash_list), len(hash_list)), int)

    for i in range(len(hash_list)):
        for j in range(len(hash_list)):
            d = diff_hash(hash_list[i], hash_list[j]);
            adj[i][j] = d
            adj[j][i] = d

    c = get_n_closest(10, file_list, adj)
    for key, value in c.items():
        print(key)
        for v in value:
            print(v)

