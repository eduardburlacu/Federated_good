#!/bin/bash


cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/ ||exit 1
cd ../../leaf/data/ ||exit 2

declare -A folders

folders["sent140"]="./preprocess.sh -s niid --sf 0.3 -k 30 -tf 0.8 -t sample"
folders["femnist"]="./preprocess.sh -s niid --sf 0.5 -k 0 -tf 0.8 -t sample"
folders["shakespeare"]="./preprocess.sh -s niid --sf 0.2 -k 0 -tf 0.8 -t sample"

# Iterate over files in the current directory
for folder in "${!folders[@]}"; do
    if [ -d "$folder" ]; then
        cd "$folder" ||exit
        echo "DOWNLOAD DATASET in: $(pwd)"
        eval "${folders[$folder]}"
        cd .. ||exit
    else
        echo "leaf/data directory '$(folder)' does not exist."
    fi
done
