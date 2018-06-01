#!/bin/python3
import sys
import os
import getopt
from collections import defaultdict
import tlsh

def usage():
    print("python3 ./diff.py [args] [in_file] [compare_files]")

try:
    opts, args = getopt.getopt(sys.argv[1:], "hdn:", ["help", "dump", "number"])
except getopt.GetoptError as err:
    print(err) # will print something like "option -a not recognized"
    usage()
    exit(2)

dump = False
n = 0
file_inputs = args

for o, a in opts:
    if o in ("-n", "--number"):
        n = int(a)
    elif o in ("-h", "--help"):
        usage()
        exit()
    elif o in ("-d", "--dump"):
        dump = True

if n <= 0:
    print("N not set, defaulting to 10")
    n = 10

if len(file_inputs) < 2:
    print("Program must contain at least 2 file inputs")
    exit(1)

if dump:
    out_data = {}
    for arg in file_inputs:
        if os.path.getsize(arg) < 256:
            print("Input file must be a minimum of 256 bytes\n")
            exit(1)

        file_hash = tlsh.hash(open(arg, 'rb').read())
        out_data[arg] = file_hash

    for elem in sorted(out_data):
        print('{},{}'.format(elem, out_data[elem]))

    exit()

if os.path.getsize(file_inputs[0]) < 256:
    print("Input file must be a minimum of 256 bytes\n")
    exit(1)

base = tlsh.hash(open(file_inputs[0], 'rb').read())
test_col = defaultdict(list)

for arg in file_inputs[1:]:
    if os.path.getsize(arg) < 256:
        print("Test file must be a minimum of 256 bytes\n")
        exit(1)

    new_hash = tlsh.hash(open(arg, 'rb').read())
    diff = tlsh.diff(base, new_hash)
    test_col[diff].append((arg, new_hash))

print("Base hash: " + str(base))

count = 0
for i in sorted(test_col)[:n]:
    print("\nDistance " + str(i))
    for elem in test_col[i]:
        print(elem[0])
        print(elem[1])
        count += 1
        if count >= n:
            exit()
