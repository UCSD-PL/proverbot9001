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

## Getting Started with RL4Proof

git submodule init CompCert && git submodule update
cd CompCert
./configure x86_64-linux
make
make data/compcert-scrape.txt && make data/scrape-test.txt
