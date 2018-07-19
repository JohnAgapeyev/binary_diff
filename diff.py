#!/bin/python3
import sys
import os
import getopt
import csv
import json
import itertools
import zipfile
import tarfile
import binwalk
import collections
from heapq import nsmallest
from collections import defaultdict
import tlsh
import numpy

from multiprocessing.dummy import Pool

def usage():
    print("python3 ./diff.py [file directory] [metadata file]")

def partition_hashes(hash_list, file_list):
    output = {}
    for h in hash_list:
        filename = file_list[hash_list.index(h)]
        quartile_range = int(h[6:10], 16)
        if quartile_range not in output:
            output[quartile_range] = [(filename, h)]
        else:
            output[quartile_range].append((filename, h))
    return output

#Values are from the TLSH paper
def convert_dist_to_confidence(d):
    if d < 30:
        return 99.99819
    elif d < 40:
        return 99.93
    elif d < 50:
        return 99.48
    elif d < 60:
        return 98.91
    elif d < 70:
        return 98.16
    elif d < 80:
        return 97.07
    elif d < 90:
        return 95.51
    elif d < 100:
        return 93.57
    elif d < 150:
        return 75.67
    elif d < 200:
        return 49.9
    elif d < 250:
        return 30.94
    elif d < 300:
        return 20.7
    else:
        return 0

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
        with tarfile.open(filename, 'r') as tar:
            for member in tar.getmembers():
                if not member or member.size < 256:
                    continue
                try:
                    meta.append(tlsh.hash(tar.extractfile(member).read()))
                    if use_binwalk:
                        for module in binwalk.scan(tar.extractfile(member).read(), signature=True, quiet=True):
                            for result in module.results:
                                meta.append(str(result.file.path))
                                meta.append(str(result.offset))
                                meta.append(str(result.description))
                except:
                    continue
    elif zipfile.is_zipfile(filename):
        try:
            with zipfile.ZipFile(filename) as z:
                for member in z.infolist():
                    if not member or member.file_size < 256:
                        continue
                    try:
                        with z.read(member) as zipdata:
                            meta.append(tlsh.hash(zipdata))
                            if use_binwalk:
                                for module in binwalk.scan(zipdata):
                                    for result in module.results:
                                        meta.append(str(result.file.path))
                                        meta.append(str(result.offset))
                                        meta.append(str(result.description))
                    except:
                        continue
        except:
            pass

    if use_binwalk:
        for module in binwalk.scan(filename, signature=True, quiet=True):
            for result in module.results:
                meta.append(str(result.file.path))
                meta.append(str(result.offset))
                meta.append(str(result.description))

    file_hash = tlsh.hash(open(filename, 'rb').read())

    if not meta:
        return file_hash
    else:
        return tlsh.hash(str.encode(file_hash + ''.join(map(str, meta))))

def lsh_json(data):
    filename = data[0]
    meta = []
    print(filename)
    if not data[1] or data[1] == None:
        pass
    else:
        stuff = [d for d in data[1] if d['filename'] == os.path.basename(filename)]
        if stuff:
            if len(stuff) >= 1:
                stuff = stuff[0]
            [meta.extend([k,v]) for k,v in stuff.items()]
            [meta.extend([k,v]) for k,v in meta[3].items()]
            del meta[3]
            [meta.extend([k,v]) for k,v in meta[-1].items()]
            del meta[-3]
            [meta.extend([k,v]) for k,v in meta[-4].items()]
            del meta[-6]

    if os.path.getsize(filename) < 256:
        raise ValueError("{} must be at least 256 bytes".format(filename))

    if tarfile.is_tarfile(filename):
        with tarfile.open(filename, 'r') as tar:
            for member in tar.getmembers():
                if not member or member.size < 256:
                    continue
                try:
                    meta.append(tlsh.hash(tar.extractfile(member).read()))
                    if use_binwalk:
                        for module in binwalk.scan(tar.extractfile(member).read(), signature=True, quiet=True):
                            for result in module.results:
                                meta.append(str(result.file.path))
                                meta.append(str(result.offset))
                                meta.append(str(result.description))
                except:
                    continue
    elif zipfile.is_zipfile(filename):
        try:
            with zipfile.ZipFile(filename) as z:
                for member in z.infolist():
                    if not member or member.file_size < 256:
                        continue
                    try:
                        with z.read(member) as zipdata:
                            meta.append(tlsh.hash(zipdata))
                            if use_binwalk:
                                for module in binwalk.scan(zipdata):
                                    for result in module.results:
                                        meta.append(str(result.file.path))
                                        meta.append(str(result.offset))
                                        meta.append(str(result.description))
                    except:
                        continue
        except:
            pass

    if use_binwalk:
        for module in binwalk.scan(filename, signature=True, quiet=True):
            for result in module.results:
                meta.append(str(result.file.path))
                meta.append(str(result.offset))
                meta.append(str(result.description))

    file_hash = tlsh.hash(open(filename, 'rb').read())

    if not meta:
        return file_hash
    else:
        return tlsh.hash(str.encode(file_hash + ''.join(map(str, meta))))

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

def parse_metadata_json(filename):
    with open(filename, 'r') as jsonfile:
        metadata = json.load(jsonfile)
        for obj in metadata:
            del obj['MD5']
            del obj['SHA1']
            del obj['SHA256']
            del obj['SHA512']
            obj['filename'] = obj['Properties'].pop('FileName')
        return metadata

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

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

def get_partition_entry(partition_hashes, new_hash):
    return partition_hashes[int(new_hash[6:10], 16)]

def get_n_closest_partial(n, hash_partition, hash_list):
    closest = {}
    for h in hash_list:
        entry = get_partition_entry(hash_partition, h)

        elem = []
        for k,v in entry:
            elem.append(diff_hash(h, v))

        print(elem)
        exit()
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
    opts, args = getopt.getopt(sys.argv[1:], "hd:m:bn:", ["help", "directory", "metadata", "binwalk", "number"])
except getopt.GetoptError as err:
    print(err) # will print something like "option -a not recognized"
    usage()
    exit(2)

directory = ""
meta = ""
use_binwalk = False
n = 10

for o, a in opts:
    if o in ("-d", "--directory"):
        directory = a
    elif o in ("-h", "--help"):
        usage()
        exit()
    elif o in ("-m", "--metadata"):
        meta = a
    elif o in ("-b", "--binwalk"):
        use_binwalk = True
    elif o in ("-n", "--number"):
        n = int(a)

if not directory:
    print("Program must be provided a file directory path")
    exit(1)

file_list = list_files(directory)
hash_list = []

if meta:
    #meta_contents = parse_metadata(meta)
    meta_contents = parse_metadata_json(meta)
else:
    meta_contents = None

hash_list = [lsh_json(x) for x in zip(file_list, itertools.repeat(meta_contents))]

stuff = partition_hashes(hash_list, file_list)

for p in stuff.items():
    print(p)

get_n_closest_partial(n, stuff, hash_list)

adj = numpy.zeros((len(hash_list), len(hash_list)), int)

for i in range(len(hash_list)):
    for j in range(len(hash_list)):
        d = diff_hash(hash_list[i], hash_list[j]);
        adj[i][j] = d
        adj[j][i] = d

c = get_n_closest(n, file_list, adj)
for key, value in c.items():
    print(key)
    for dist, name in value:
        print(convert_dist_to_confidence(dist), dist, name)

