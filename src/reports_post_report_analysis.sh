#!/usr/bin/env bash
set -x
set -e

job_name=$1
echo "Analyzing reports"

echo -n > output/rl_$job_name/Compare_reports.txt


folders_string_baseline=$(find output/baselines -name "*-search-report*")
folders_baseline=()
for folder in $folders_string_baseline; do
    folders_baseline+=($folder)
done

folders_string_rl=$(find output/rl_$job_name -name "*-search-report*")
folders_rl=()
for folder in $folders_string_rl; do
    folders_rl+=($folder)
done

for ((i=0; i<${#folders_rl[@]}; i++)); do
    for ((j=i+1; j<${#folders_rl[@]}; j++)); do
        folder1="${folders_rl[i]}"
        folder2="${folders_rl[j]}"
        echo "Comparing {$folder1} with {$folder2}"
        folder1_name=$(basename $folder1)
        folder2_name=$(basename $folder2)
        printf "\n --------- Comparing ${folder1} with ${folder2} --------------- \n" >> output/rl_$job_name/Compare_reports.txt
        srun python src/compare_steps.py "$folder1" "$folder2" --a-name=$folder1_name  --b-name=$folder2_name  >> output/rl_$job_name/Compare_reports.txt
    done
done

#LLMs are amazing at debugging bash code
for ((i=0; i<${#folders_baseline[@]}; i++)); do
    for ((j=0; j<${#folders_rl[@]}; j++)); do
        folder1="${folders_baseline[i]}"
        folder2="${folders_rl[j]}"
        echo "Comparing {$folder1} with {$folder2}"
        folder1_name=$(basename $folder1)
        folder2_name=$(basename $folder2)
        printf "\n --------- Comparing ${folder1} with ${folder2} --------------- \n" >> output/rl_$job_name/Compare_reports.txt
        srun --time="00:05:00" python src/compare_steps.py "$folder1" "$folder2" --a-name=$folder1_name  --b-name=$folder2_name  >> output/rl_$job_name/Compare_reports.txt
    done
done