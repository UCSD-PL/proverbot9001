# Proverbot9001
![Proverbot logo](proverbotlogo-01.png)
A bot for proving.

You can find the paper and talk video at [our website](https://proverbot9001.ucsd.edu).

## Prerequisites

You'll need to install `git`, `opam`, `graphviz`, `libgraphviz-dev`,
`python3.7`, `python3.7-dev` and `python3.7-pip` to run Proverbot.

If you're running Linux, all three can be generally found in your package repositories.
If that Linux is ubuntu, you'll have to first run:
```
sudo apt install software-common-properties
sudo add-apt-repository ppa:avsm/ppa
sudo apt update
```
before you can install OPAM.

If you're running OS X, you can find these packages in Homebrew.

If you're on Windows, follow:
https://gitforwindows.org/
https://fdopen.github.io/opam-repository-mingw/installation/
https://graphviz.gitlab.io/_pages/Download/Download_windows.html
https://www.python.org/downloads/windows/

or use Windows Subsystem for Linux

## Getting Started

The first thing you'll need to do is clone the repository (download the code).

I recommend that you do this over ssh. Open a terminal (windows users
can use "Bash on Ubuntu on Windows"), move to the directory you want
to work in, and run:

```
git clone git@github.com:UCSD-PL/proverbot9001.git
```

That should give you a new folder called `proverbot9001` with all the
code inside. Move into this new directory:

```
cd proverbot9001
```

And run this command to install the dependencies

```
make setup
```

this step will take a while, and might involve having to type `y` a
few times.

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
make report
```

Which will generate some html in the `reports` directory.

You should be able to check out the results by opening a web browser
and navigating to the `reports` directory in the project.

Once you've done that, you might like to try training from scratch
with `make scrape` and `make train`, or [writing your own
predictor](predictor.md).
