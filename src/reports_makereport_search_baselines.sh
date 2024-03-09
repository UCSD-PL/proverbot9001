#!/usr/bin/env bash



eval $(opam env)

echo "Running baseline searches"

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/baselines/DFS-search-report 

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/baselines/DFS-search-report_128 --max-steps=128

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/baselines/DFS-search-report_256 --max-steps=256

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/baselines/DFS-search-report_512 --max-steps=512

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/baselines/DFS-search-report_1024 --max-steps=1024

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/baselines/DFS-search-report_ms128_dep4 --max-steps=128 --search-depth=4

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/baselines/DFS-search-report_ms128_dep6 --max-steps=128 --search-depth=6

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/baselines/DFS-search-report_ms128_dep8 --max-steps=128 --search-depth=8

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --search-type=dfs --output=output/baselines/DFS-search-report_ms128_dep2 --max-steps=128 --search-depth=2

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --scoring-function=const --search-type=astar --output=output/baselines/Astar-search-report_const  --max-steps=128

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --scoring-function=certainty --search-type=best-first --output=output/baselines/BestFirst-search-report_cert  --max-steps=128

python ./src/search_file_cluster.py -p cpu --mem=8G --hard-depth-limit=20 --prelude=./CompCert --num-workers=32 --weightsfile=data/polyarg-weights-develop.dat -j1 compcert_projs_splits.json --scoring-function=const --search-type=best-first --output=output/baselines/BestFirst-search-report_const  --max-steps=128