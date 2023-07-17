#!/bin/bash

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/ ||exit 1
cd ../../leaf/data/ ||exit 2

declare -A folders

if [[ "$1" == "full" ]]; then
    folders["celeba"]="./preprocess.sh -s niid --sf 1.0 -k 5 -t sample"
    folders["femnist"]="./preprocess.sh -s niid --sf 1.0 -k 0 -t sample"
    folders["sent140"]="./preprocess.sh -s niid --sf 1.0 -k 0 -t sample"
    folders["shakespeare"]="./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8"
else
    folders["celeba"]="./preprocess.sh -s niid --sf 0.03 -k 5 -t sample"
    folders["femnist"]="./preprocess.sh -s niid --sf 0.03 -k 0 -t sample"
    folders["sent140"]="./preprocess.sh -s niid --sf 0.05 -k 3 -t sample"
    folders["shakespeare"]="./preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8"
fi

# Iterate over files in the current directory
for folder in "${!folders[@]}"; do
    if [ -d "$folder" ]; then
        cd "$folder" ||exit
        if [ ! -d "data" ]; then
            echo "Action to be taken in: $(pwd)"
            echo "${folders[$folder]}"
        fi
        cd .. ||exit
    else
        echo "leaf/data directory '$(folder)' does not exist."
    fi
done
