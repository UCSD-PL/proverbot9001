# TODO

## next actions
- [x] switch over to non-dev-container version (for torch to run locally)
- [ ] setup automatic history tracking
- [x] remove all submodules
- [x] run 'make setup'
- [ ] look into issues listed below and try to fix the pygraphviz error
- [ ] run 'make scrape'
- [ ] try running inference on dev container (you don't need GPUs for it)
      - could use cog to run proverbot in a container (https://github.com/replicate/cog)
- [ ] writeup what I did to get it working on M1 so we can add it to the README.

## someday/maybe


## notes/ideas

Proof state as context
Llm guess relevant propositions as context

### Dockerizing
pytorch needs to run natively if it uses the GPU, as M1 mac gpus are not accessible via docker
however, the specifics of opam, rust, etc are on the CPU and are a total pain. can we run those in docker?
i.e. mine the data in docker, run inference natively


### Getting proverbot working
constraints:
1. I can’t use a docker container because torch doesn’t work with m1 macs thru docker
2. coq 8.17.0 doesn’t work with proverbot because compcert uses omega
  - https://stackoverflow.com/questions/72298228/coq-make-failing-on-omega


https://github.com/pygraphviz/pygraphviz/issues/342
pip install --global-option=build_ext \
--global-option="-I/opt/homebrew/include/" \
--global-option="-L/opt/homebrew/lib/" --force-reinstall -v "pygraphviz==1.6"

checking out compcert @ master
using coq 8.13.0

in compcert
./configure aarch64-macos

I basically had to hand execute setup.sh

### Running `make search-report` gives an error

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