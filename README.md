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
   (website)[https://www.python.org/].
2. Make sure pip, the python package manager, is available, by running
   in your terminal: `python -m ensurepip`.
3. Install Homebrew from their (website)[https://brew.sh/].
4. Install wget, git, opam, rustup, GNU awk, and graphviz through Homebrew:
   `brew install wget git opam rustup-init gawk graphviz && rustup-init`
5. On MacOS, you'll need to install pygraphviz differently, so do that
now: ` pip install --global-option=build_ext
--global-option="-I/usr/local/include/"
--global-option="-L/usr/local/lib/" pygraphviz`

### Linux
1. Check your python version with `python --version` in the
   terminal. If your version is older than Python 3.7, or the python
   command isn't found, install python through your package manager.
   a. On Ubuntu, that's:
   ```
   sudo apt install python3 python3-dev python3-pip
   ```
2. Make sure pip, the python package manager, is available, by running
   in your terminal: `python -m ensurepip`.
3. Install git, opam, rustup, and graphviz using your package manager.
   a. On Ubuntu, that's:
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

I recommend that you do this over ssh. To clone github projects using
git@ urls (over ssh), you'll need to follow the instructions on (this
github
page)[https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent]. Then,
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

this command to install the dependencies:

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

Once that's finished, you're ready to start running the tool!

Running Proverbot9001 requires training on existing proof
files. Training takes a while, and usually you need some pretty
advanced hardware. So to quickly get started, we'll download
pre-trained weights instead:

```
make download-weights
```

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
