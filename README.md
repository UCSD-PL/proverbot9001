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
5. On newer MacOS systems, homebrew installs into `/opt/homebrew` not
   `/usr/local`, so run:
   `export CPATH=/opt/homebrew/include && export LIBRARY_PATH=/opt/homebrew/lib`.

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

## Getting Started

The first thing you'll need to do is clone the repository (download the code).

### Cloning the repository (downloading the project)
I recommend that you do this over ssh. To clone github projects using
git@ urls (over ssh), you'll need to follow the instructions on [this
github
page](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Then,
open a terminal (windows users can use "Bash on Ubuntu on Windows"),
move to the directory you want to work in, and run:

```
git clone git@github.com:UCSD-PL/proverbot9001.git
```

Alternatively, you can clone over https without setting up your github
keys, with this command:
```
git clone https://github.com/UCSD-PL/proverbot9001.git
```

That should give you a new folder called `proverbot9001` with all the
code inside. Move into this new directory:

```
cd proverbot9001
```

### Create a python virtual environment

Next, you'll need to create a python virtual environment to work
in. This is a good idea in general, but is also required for maturin
to work properly.

```
python -m venv proverbot-env
```

Whenever you want to work with Proverbot from a new shell, you'll have
to first activate this environment:

```
source proverbot-env/bin/activate
```

Also do that now.

### Install python and rust dependencies

On MacOS, you'll need to install pygraphviz differently, so do that
now: ` pip install --global-option=build_ext
--global-option="-I/usr/local/include/"
--global-option="-L/usr/local/lib/" pygraphviz`

Then, no matter your platform, run this command to install the
dependencies:

```
make setup
```

this step will take a while, and might involve having to type `y` a
few times.

If this step fails, and part of the error message near the bottom says:
```
pygraphviz/graphviz_wrap.c:2711:10: fatal error: graphviz/cgraph.h: No such file or directory
 2711 | #include "graphviz/cgraph.h"
      |          ^~~~~~~~~~~~~~~~~~~
compilation terminated.
error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```
then python needs help finding your graphviz installation. Check out this github issue: https://github.com/pygraphviz/pygraphviz/issues/155, and possibly this one: https://github.com/pypa/setuptools/issues/2740
.

### Download the pre-trained weights

Running Proverbot9001 requires training on existing proof
files. Training takes a while, and usually you need some pretty
advanced hardware. So to quickly get started, we'll download
pre-trained weights instead:

```
make download-weights
```

### Running the tool

Now you can run Proverbot9001:

```
make search-report
```

Which will generate some html in the `reports` directory.

You should be able to check out the results by opening a web browser
and navigating to the `reports` directory in the project.

Once you've done that, you might like to try training from scratch
with `make scrape` and `make train`, or [writing your own
predictor](predictor.md).

### Results Reports

The latest Proverbot9001 results can be found
[here](https://proverbot9001.ucsd.edu/reports/2024-01-26T19d27d24-0700+a13e20f06b2535d9abd20853749b0709b280e054/index.html).
This is a report in the format generated by Proverbot9001's `make
search-report`; you can click on individual files to see exactly which
theorems were proven by Proverbot9001 and what the proofs are.

You can also view the results that were used in the 2020 MAPL paper
(linked above)
[here](https://proverbot9001.ucsd.edu/reports/2019-11-20T18d38d02-0700+cd762eb9e7e6e44153bd766654727a36a3dcad0b/report.html).
However, these results are *not up-to-date*. If you use these results
for a comparison or otherwise write about them, please add a footnote
or other comment noting that these results are *significantly* worse
than the latest Proverbot9001 results.
