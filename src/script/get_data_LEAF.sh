#!/bin/bash

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/ ||exit
cd ../../leaf/data/ ||exit

# Iterate over files in the current directory
for file in *; do
    if [ -d "$file" ]; then
        cd "$file"
        echo "Action to be taken in: $(pwd)"
        echo "./preprocess.sh -s niid --sf 0.03 -k 3 -t sample"
        cd ..
    fi
done
