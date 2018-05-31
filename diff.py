#!/bin/python3
import sys
import os
from collections import defaultdict
import tlsh

if len(sys.argv) < 3:
    print("Requires a minimum of two files as arguments")
    exit(1)

if os.path.getsize(sys.argv[1]) < 256:
    print("Input file must be a minimum of 256 bytes\n")
    exit(1)

base = tlsh.hash(open(sys.argv[1], 'rb').read())
n = 10
test_col = defaultdict(list)

for arg in sys.argv:
    if arg == sys.argv[0] or arg == sys.argv[1]:
        continue

    if os.path.getsize(arg) < 256:
        print("Test file must be a minimum of 256 bytes\n")
        exit(1)

    new_hash = tlsh.hash(open(arg, 'rb').read())
    diff = tlsh.diff(base, new_hash)
    test_col[diff].append((arg, new_hash))

print("Base hash: " + str(base))

for i in sorted(test_col)[:n]:
    print("Distance " + str(i))
    for elem in test_col[i]:
        print(elem[0])
        print(elem[1])
