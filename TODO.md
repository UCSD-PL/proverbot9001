# TODO

## next actions
- [x] switch over to non-dev-container version (for torch to run locally)
- [x] setup automatic history tracking
- [x] remove all submodules
- [x] run 'make setup'
- [x] setup automatic history tracking on bash
- [x] look into issues listed below and try to fix the pygraphviz error
- [ ] run 'make scrape' 
- [x] try running inference on dev container (you don't need GPUs for it)
      - could use cog to run proverbot in a container (https://github.com/replicate/cog)
- [ ] writeup what I did to get it working on M1 so we can add it to the README.
- [ ] output embeddings from hypothesis relevance predictor on test set
- [ ] read [predictor.md](predictor.md)

## someday/maybe
- [ ] setup a pyproject.toml

## notes/ideas

Proof state as context
Llm guess relevant propositions as context

### Dockerizing
pytorch needs to run natively if it uses the GPU, as M1 mac gpus are not accessible via docker
however, the specifics of opam, rust, etc are on the CPU and are a total pain. can we run those in docker?
i.e. mine the data in docker, run inference natively


### Getting proverbot working on an M1 mac
constraints:
1. I can’t use a docker container because torch doesn’t work with m1 macs thru docker
2. coq 8.17.0 doesn’t work with proverbot because compcert uses omega
  - https://stackoverflow.com/questions/72298228/coq-make-failing-on-omega


https://github.com/pygraphviz/pygraphviz/issues/342
pip install --global-option=build_ext \
--global-option="-I/opt/homebrew/include/" \
--global-option="-L/opt/homebrew/lib/" --force-reinstall -v "pygraphviz==1.6"

checking out compcert @ master
- rm `./exportclight/Clightdefs.v` from compcert test files


in setup.sh, change `opam pin add coq 8.10.2` to `opam pin add coq 8.13.0`

in compcert
./configure aarch64-macos

I basically had to hand execute setup.sh

#### Running `make search-report` gives an error

```
(export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH ; cat data/compcert-test-files.txt | cat | \
        xargs ./src/proverbot9001.py search-report -j 16 --weightsfile=data/polyarg-weights.dat --prelude=./CompCert --search-depth=6 --search-width=5 -P )
Traceback (most recent call last):
  File "/Users/sakekasi/school/1 phd/spring-0/proverbot9001/./src/proverbot9001.py", line 27, in <module>
    import search_file
  File "/Users/sakekasi/school/1 phd/spring-0/proverbot9001/src/search_file.py", line 49, in <module>
    import search_report
  File "/Users/sakekasi/school/1 phd/spring-0/proverbot9001/src/search_report.py", line 48, in <module>
    from search_worker import get_file_jobs
  File "/Users/sakekasi/school/1 phd/spring-0/proverbot9001/src/search_worker.py", line 15, in <module>
    from search_strategies import best_first_proof_search, bfs_beam_proof_search, dfs_proof_search_with_graph
  File "/Users/sakekasi/school/1 phd/spring-0/proverbot9001/src/search_strategies.py", line 14, in <module>
    import pygraphviz as pgv
  File "/Users/sakekasi/school/1 phd/spring-0/proverbot9001/proverbot-env/lib/python3.9/site-packages/pygraphviz/__init__.py", line 56, in <module>
    from .agraph import AGraph, Node, Edge, Attribute, ItemAttribute, DotError
  File "/Users/sakekasi/school/1 phd/spring-0/proverbot9001/proverbot-env/lib/python3.9/site-packages/pygraphviz/agraph.py", line 20, in <module>
    from . import graphviz as gv
  File "/Users/sakekasi/school/1 phd/spring-0/proverbot9001/proverbot-env/lib/python3.9/site-packages/pygraphviz/graphviz.py", line 13, in <module>
    from . import _graphviz
ImportError: dlopen(/Users/sakekasi/school/1 phd/spring-0/proverbot9001/proverbot-env/lib/python3.9/site-packages/pygraphviz/_graphviz.cpython-39-darwin.so, 0x0002): symbol not found in flat namespace '_Agdirected'
make: *** [search-report] Error 1
```

pip install --upgrade --global-option=build_ext \
--global-option="-I$(brew --prefix graphviz)/include/" \
--global-option="-L$(brew --prefix graphviz)/lib/" --force-reinstall pygraphviz

3 issues:
- https://github.com/pygraphviz/pygraphviz/issues/398
- https://github.com/pygraphviz/pygraphviz/issues/342
- https://github.com/pygraphviz/pygraphviz/issues/400

#### working around `make search-report`
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH ; \
  cat data/compcert-test-files.txt | cat | \
  xargs ./src/proverbot9001.py search-report -j 16 --weightsfile=data/polyarg-weights.dat --prelude=./CompCert --search-depth=6 --search-width=5 -P

opam's env string is currently using fish, not bash since my shell is fish.
maybe the right move here is to use bash on linux, and use torch with cpu not mps

commented out `pygraphviz` related stuff in `rgraph.py` and `search_strategies.py`

### trying again to get it working on the dev container

Because it seems really overfit to bash and linux. Seems like getting closer to a linux environment is the right move.

```
python -m venv proverbot-env
source proverbot-env/bin/activate
pip install -r requirements.txt 
pip install --upgrade pip
make setup
```

got this error during make setup:
```
[ERROR] Sandboxing is not working on your platform debian:
        "~/.opam/opam-init/hooks/sandbox.sh build sh -c echo SUCCESS
        >$TMPDIR/opam-sandbox-check-out && cat $TMPDIR/opam-sandbox-check-out; rm -f
        $TMPDIR/opam-sandbox-check-out" exited with code 1 "bwrap: No permissions to create new
        namespace, likely because the kernel does not allow non-privileged user namespaces. See
        <https://deb.li/bubblewrap> or <file:///usr/share/doc/bubblewrap/README.Debian.gz>."
```
This is fine, we're running in a container

```
git submodule deinit -f .
git submodule update --init
make setup
```

everything in setup.sh runs fine except 
`setup-compcert`

doing that by hand

```
eval $(opam env) # setup switch
./configure aarch64-linux
```

../src/patch_compcert.sh fails with
sed: preserving permissions for ‘/workspace/src/../CompCert/backend/seduqi2mA’: Permission denied

make download-weights works

make search-report
- maxes out CPU and runs *very slow*

uncommented util.use_cuda = False in search_file.py. that made it a lot faster

### running make scrape

- currently, it's failing all 2075 files
  - `Didn't find _CoqProject or Make for ../coq-projects` 
  - `Cannot find a physical path bound to logical path matching suffix bbv.`
  - `Unable to locate library Cabs`
  - `Cannot find a physical path bound to logical path matching suffix Metalib.`

actually, seems like it's running a lot of sucessful scrapes [like this one](coq-projects/hedges/hedges.v.scrape)
