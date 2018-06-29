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

### Python 3.6
You'll also need Python 3.6 (well, an earlier version is possible to
use, but requires hacking the makefile a bit).

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
