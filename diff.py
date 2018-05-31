#!/bin/python3
import sys
import tlsh

if len(sys.argv) < 3:
    print("Requires a minimum of two files as arguments")
    exit(1)

if len(open(sys.argv[1], 'rb').read()) < 256:
    print("Input file must be a minimum of 256 bytes\n")
    exit(1)

base = tlsh.hash(open(sys.argv[1], 'rb').read())
file_hashes = {}
closest_files = []
min_dist = (2**16) - 1

for arg in sys.argv:
    if arg == sys.argv[0] or arg == sys.argv[1]:
        continue

    if len(open(arg, 'rb').read()) < 256:
        print("Test file must be a minimum of 256 bytes\n")
        exit(1)

    new_hash = tlsh.hash(open(arg, 'rb').read())
    file_hashes[arg] = new_hash
    diff = tlsh.diff(base, new_hash)
    if diff < min_dist:
        min_dist = diff
        closest_files = [arg]
    elif diff == min_dist:
        closest_files.insert(arg)

print(closest_files)
print("Base hash: " + str(base))
print(file_hashes)
