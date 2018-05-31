#!/bin/bash
cd tlsh
./make.sh
cd py_ext
python ./setup.py build
sudo python ./setup.py install
cd ../Testing
./python_test.sh
