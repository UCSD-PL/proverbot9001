# Proverbot9001
![Proverbot logo](proverbotlogo-01.png)
A bot for proving.

## Prerequisites

### OPAM
To build and run this project, you'll need to already have opam
installed. If you're on a linux machine, this can be accomplished
through your package manager with:

(Ubuntu)
```
sudo apt-get install opam
```

(Fedora)
```
sudo yum install opam
```

(Arch Linux)
```
sudo pacman -S opam
```

(Gentoo)
```
sudo emerge --ask opam
```

If you're on a Windows machine,
follow
[these](https://www.cs.umd.edu/class/spring2018/cmsc330/ocamlInstallationGuide.pdf) instructions.

### Python
You'll also need Python 3.5 or 3.6, and Pip 3. This is probably already installed
on your machine, you can test by running

```
python3 --version
pip3 --version
```

If either of those commands show a python version lower than 3.5, or
they report "command not found", you'll need to install them. You can
do this much like installing OPAM, through your local package manager. On Ubuntu:

```
sudo apt-get install python3 python3-pip
```

### Graphviz
The proof search graphs are produced using graphviz
```
sudo apt-get install graphviz
```

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

Once you've done that, you might like to
try [writing your own predictor](predictor.md).
