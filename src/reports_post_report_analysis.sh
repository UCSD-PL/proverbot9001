#!/usr/bin/env bash

echo "Analyzing reports"


folders_string=$(find output/ -name "*-search-report*")
echo -n > output/rl/Compare_reports.txt

folders=()
for folder in $folders_string; do
    folders+=($folder)
done

#LLMs are amazing at debugging bash code
for ((i=0; i<${#folders[@]}; i++)); do
    for ((j=i+1; j<${#folders[@]}; j++)); do
        folder1="${folders[i]}"
        folder2="${folders[j]}"
        if [[ $folder2 != *"baseline"* ]]; then
            printf "\n --------- Comparing ${folder1} with ${folder2} --------------- \n" >> output/rl/Compare_reports.txt
            srun python src/compare_steps.py "$folder1" "$folder2" >> output/rl/Compare_reports.txt
        fi
    done
done