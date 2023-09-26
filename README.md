[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generating-correctness-proofs-with-neural/automated-theorem-proving-on-compcert)](https://paperswithcode.com/sota/automated-theorem-proving-on-compcert?p=generating-correctness-proofs-with-neural)

# Proverbot9001
![Proverbot logo](proverbotlogo-01.png)
A bot for proving.

You can find the paper and talk video at [our website](https://proverbot9001.ucsd.edu).

## Prerequisites

### MacOS

1. Check your python version with `python --version` in the
   terminal. If your version is older than Python 3.7, or the python
   command isn't found, install python through the python
   [website](https://www.python.org/).
2. Make sure pip, the python package manager, is available, by running
   in your terminal: `python -m ensurepip`.
3. Install Homebrew from their [website](https://brew.sh/).
4. Install wget, git, opam, rustup, GNU awk, and graphviz through Homebrew:
   `brew install wget git opam rustup-init gawk graphviz && rustup-init`

### Linux
1. Check your python version with `python --version` in the
   terminal. If your version is older than Python 3.7, or the python
   command isn't found, install python through your package manager.
   1. On Ubuntu, that's:
   ```
   sudo apt install python3 python3-dev python3-pip
   ```
2. Make sure pip, the python package manager, is available, by running
   in your terminal: `python -m ensurepip`.
3. Install git, opam, rustup, and graphviz using your package manager.
   1. On Ubuntu, that's:
   ```
   sudo apt install software-properties-common
   sudo add-apt-repository ppa:avsm/ppa
   sudo apt update
   sudo apt install git opam rustup graphviz libgraphviz-dev
   ```

### Windows
Windows support is more experimental, but you can try installing
prereqs through:

https://gitforwindows.org/

https://fdopen.github.io/opam-repository-mingw/installation/

https://graphviz.gitlab.io/_pages/Download/Download_windows.html

https://www.python.org/downloads/windows/

or use Windows Subsystem for Linux
### Setting up environments for distributed RL on Unity
1. ```modules load openmpi/4.1.3+cuda11.6.2```
2. ```git clone --recursive git@github.com:pytorch/pytorch.git```
3. ```If using conda: `conda install cmake ninja```
4.
```cd pytorch
pip install -r requirements.txt
pip install mkl mkl-include
pip install pytorch magma-cuda117
git submodule sync && git submodule update --init --recursive
```
5. If using conda: ```export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}```
6. ```CMAKE_C_COMPILER=$(which mpicc) CMAKE_CXX_COMPILER=$(which mpicxx) python setup.py develop```
7. **Important** make sure you run ```module load opam/2.1.2 graphviz/2.49.0+py3.8.12 openmpi/4.1.3+cuda11.6.2``` before each run. 


## Getting Started with RL4Proof
```
git submodule init CompCert && git submodule update
cd CompCert
./configure x86_64-linux
make
make data/compcert-scrape.txt && make data/scrape-test.txt
```
Note the make commands are to be ran under CompCert directory. If you are using a Unity cluster, ```srun -c8``` and use ```-j8``` arguments for make.

### Troubleshooting
- Make sure you are using ```Coq 8.10.2```.
- If anything went wrong and you need to re-make the CompCert, run ```make clean``` under CompCert directory and repeat the above steps again.


## Running the script
### Checklist
1. Make sure CompCert making is done.
2. Have ```data/polyarg-weights-develop.dat``` and ```data/term2vec-weights-59.dat``` included in your local repo.
### Generating tasks
Run
```
python src/gen_rl_tasks.py --prelude=CompCert \
      --supervised-weights=data/polyarg-weights-develop.dat -o rl_train_jobs.json compcert_projs_splits.json
```
To generate training tasks. To generate test tasks, specify ```--data-partition='test'```. To run faster, use ```src/gen_rl_tasks_cluster.py``` instead.
### Filter data by length
By default, ```gen_rl_tasks.py``` extracts tasks by making 16 predictions at each state, and seeing if the solution tactic matches any of them. Also 
by default, ```rl.py``` only makes 5 predictions at each state to choose between. Therefore, if using both defaults, make sure to filter task length to 
at least 5 and prediction width to at most 5.
You can use the jq tool to filter task length and width.
For example, to filter as specified above, run the following command:
```
jq -c "select(.largest_prediction_idx <= 5 and .target_length >= 5)" rl_train_jobs.json > rl_train_jobs_len5_wid5.json
```
### Fill in task curriculum
Run 
```
python src/fill_in_task_curriculum.py $INPUT_FILE $OUTPUT_FILE
```
to fill in task currriculum. INPUT_FILE should be the output of the jq tool. The output file is used for the tasks-file parameter in the next step.
### Run Reinforcement Learning Script
```
python src/rl.py --supervised-weights=data/polyarg-weights-develop.dat --coq2vec-weights=data/term2vec-weights-59.dat compcert_projs_splits.json \
         --tasks-file=rl_train_jobs.json --prelude=./CompCert --backend=serapi --allow-partial-batches \
         --learning-rate=0.0001 -n10 -o data/rl_weights-compcert-5.dat -s5
```
You may specify the number of episode to be ran to by passing that to  ```-n```.  

### Running Distributed RL

```
python src/distributed_rl.py --mem=8G --num-actors=2 --supervised-weights=data/polyarg-weights-develop.dat --coq2vec-weights=../coq2vec/term2vec-weights-59.dat compcert_projs_splits.json --prelude=./CompCert --backend=serapi --gamma=0.7 -s7 -p5 --learning-rate=0.000005 -n24 -o data/rl_weights_distributed_test.dat --tasks-file=single_task.json --resume=yes
```
