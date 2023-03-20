
SHELL=/usr/bin/env bash

ENV_PREFIX=export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$$LD_LIBRARY_PATH

NTHREADS?=16
FLAGS?=
HIDDEN_SIZE=512

SITE_SERVER=goto
SITE_DIR=~alexss/proverbot9001-site
SITE_PATH=$(SITE_SERVER):$(SITE_DIR)

ifeq ($(NUM_FILES),)
HEAD_CMD=cat
else
HEAD_CMD=head -n $(NUM_FILES)
endif

ifneq ($(MESSAGE),)
FLAGS+=-m "$(MESSAGE)"
endif
REPORT?="report"
WEIGHTSFILE='data/polyarg-weights.dat'
TESTFILES=$(patsubst %, CompCert/%, $(shell cat data/compcert-test-files.txt))
COMPCERT_TRAIN_FILES=$(patsubst %, CompCert/%, $(shell cat data/compcert-train-files.txt))
TESTSCRAPES=$(patsubst %,%.scrape,$(TESTFILES))
CC_TRAIN_SCRAPES=$(patsubst %,%.scrape,$(COMPCERT_TRAIN_FILES))

.PHONY: scrape report setup static-report dynamic-report search-report

all: scrape report

setup:
	./src/setup.sh

data/compcert-scrape.txt: $(CC_TRAIN_SCRAPES)
	cat $(CC_TRAIN_SCRAPES) > $@

scrape:
	cp data/scrape.txt data/scrape.bkp 2>/dev/null || true
	cd src && \
	cat ../data/coq-projects-train-files.txt | $(HEAD_CMD) | \
	xargs python3.7 scrape.py $(FLAGS) -v -c -j $(NTHREADS) --output ../data/scrape.txt \
				        		 --prelude ../coq-projects
data/scrape-test.txt: $(TESTSCRAPES)
	cat $(TESTSCRAPES) > $@
CompCert/%.scrape: CompCert/%
	python3 src/scrape.py $(FLAGS) -c -j 1 --prelude=./CompCert $* -o /dev/null || true

report: $(TESTSCRAPES)
	($(ENV_PREFIX) ; cat data/compcert-test-files.txt | $(HEAD_CMD) | \
	xargs ./src/proverbot9001.py static-report -j $(NTHREADS) --weightsfile=data/polyarg-weights.dat --prelude ./CompCert $(FLAGS))

compcert-train: data/compcert-scrape.txt
	(cd dataloader/dataloader-core && maturin develop -r)
	./src/proverbot9001.py train polyarg data/compcert-scrape.txt data/polyarg-weights.dat --load-tokens=tokens.txt --context-filter="(goal-args+((tactic:induction+tactic:destruct)%numeric-args)+hyp-args+rel-lemma-args)%maxargs:1%default" $(FLAGS) #--hidden-size $(HIDDEN_SIZE)
train:
	./src/proverbot9001.py train polyarg data/scrape.txt data/polyarg-weights.dat --load-tokens=tokens.txt --save-tokens=tokens.pickle --context-filter="(goal-args+((tactic:induction+tactic:destruct)%numeric-args)+hyp-args+rel-lemma-args)%maxargs:1%default" $(FLAGS) #--hidden-size $(HIDDEN_SIZE)

static-report: #$(TESTSCRAPES)
	($(ENV_PREFIX) ; cat data/compcert-test-files.txt | $(HEAD_CMD) | \
	xargs ./src/proverbot9001.py static-report -j $(NTHREADS) --weightsfile=data/polyarg-weights.dat --context-filter="goal-changes" --prelude=./CompCert $(FLAGS))

dynamic-report:
	($(ENV_PREFIX) ; cat data/compcert-test-files.txt | $(HEAD_CMD) | \
	xargs ./src/proverbot9001.py dynamic-report -j $(NTHREADS) --weightsfile=data/polyarg-weights.dat --context-filter="goal-changes" --prelude=./CompCert $(FLAGS))

search-report:
	($(ENV_PREFIX) ; cat data/compcert-test-files.txt | $(HEAD_CMD) | \
	xargs ./src/proverbot9001.py search-report -j $(NTHREADS) --weightsfile=data/polyarg-weights.dat --prelude=./CompCert --search-depth=6 --search-width=5 -P $(FLAGS))

search-test:
	./src/proverbot9001.py search-report -j $(NTHREADS) --weightsfile=data/polyarg-weights.dat --prelude=./CompCert --search-depth=5 --search-width=5 -P --use-hammer -o=test-report --debug ./backend/Locations.v $(FLAGS)

scrape-test:
	cp data/scrape.txt data/scrape.bkp 2>/dev/null || true
	cd src && \
	cat ../data/coq-projects-train-files.txt | tail -n +320 | head -n 50 | \
	xargs python3.7 scrape.py $(FLAGS) -v -c -j $(NTHREADS) --output ../data/scrape.txt \
				        		 --prelude ../coq-projects

INDEX_FILES=index.js index.css build-index.py

reports/index.css: reports/index.scss
	sass $^ $@

update-index: $(addprefix reports/, $(INDEX_FILES))
	rsync -avz $(addprefix reports/, $(INDEX_FILES)) $(SITE_PATH)/reports/
	ssh goto 'cd $(SITE_DIR)/reports && \
		  python3 build-index.py'

publish:
	$(eval REPORT_NAME := $(shell ./reports/get-report-name.py $(REPORT)/))
	mv $(REPORT) $(REPORT_NAME)
	chmod +rx $(REPORT_NAME)
	tar czf report.tar.gz $(REPORT_NAME)
	rsync -avzP report.tar.gz $(SITE_PATH)/reports/
	ssh goto 'cd ~alexss/proverbot9001-site/reports && \
                  tar xzf report.tar.gz && \
                  rm report.tar.gz && \
		  chgrp -Rf proverbot9001 $(REPORT_NAME) $(INDEX_FILES) && \
		  chmod -Rf g+rw $(REPORT_NAME) $(INDEX_FILES) || true'
	mv $(REPORT_NAME) $(REPORT)
	$(MAKE) update-index

publish-weights:
	rsync -avzP $(WEIGHTSFILE) goto:proverbot9001-site/downloads/weights-`git branch --show-current`-`date -I`.dat
	ssh goto cp proverbot9001-site/downloads/weights-`git branch --show-current`-`date -I`.dat proverbot9001-site/downloads/weights-`git branch --show-current`-latest.dat

download-weights:
	curl -L -o data/polyarg-weights.dat https://proverbot9001.ucsd.edu/downloads/weights-`git branch --show-current`-latest.dat

publish-depv:
	opam info -f name,version menhir ocamlfind ppx_deriving ppx_import cmdliner core_kernel sexplib ppx_sexp_conv camlp5 | awk '{print; print ""}' > known-good-dependency-versions.md

clean:
	rm -rf report-*
	rm -f log*.txt

clean-scrape:
	for file in `find CompCert -name "*.scrape"` ; do \
            mv $$file $$file.bkp; \
        done

clean-test-scrape:
	rm -r $(TESTSCRAPES)

clean-progress:
	fd '.*\.v\.lin' CompCert | xargs rm -f
	fd '.*\.scrape' CompCert | xargs rm -f
