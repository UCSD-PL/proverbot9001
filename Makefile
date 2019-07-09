
SHELL=/usr/bin/env bash

ENV_PREFIX=export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$$LD_LIBRARY_PATH

NTHREADS=16
FLAGS=
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
REPORT="report"

.PHONY: scrape report setup

all: scrape report

setup:
	./src/setup.sh && $(MAKE) publish-depv

scrape:
	cp data/scrape.txt data/scrape.bkp 2>/dev/null || true
	cd src && \
	cat ../data/compcert-train-files.txt | $(HEAD_CMD) | \
	xargs python3 scrape.py $(FLAGS) -v -c -j $(NTHREADS) --output ../data/scrape.txt \
				        		 --prelude ../CompCert
report:
	($(ENV_PREFIX) ; cat data/compcert-test-files.txt | $(HEAD_CMD) | \
	xargs ./src/proverbot9001.py static-report -j $(NTHREADS) --weightsfile=data/polyarg-weights.dat --prelude ./CompCert $(FLAGS))

train:
	./src/proverbot9001.py train polyarg data/scrape.txt data/polyarg-weights.dat $(FLAGS) #--hidden-size $(HIDDEN_SIZE)

test:
	./src/proverbot9001.py report -j $(NTHREADS) --prelude ./CompCert ./lib/Parmov.v --predictor=ngramclass

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
	rsync -avz report.tar.gz $(SITE_PATH)/reports/
	ssh goto 'cd ~alexss/proverbot9001-site/reports && \
                  tar xzf report.tar.gz && \
                  rm report.tar.gz && \
		  chgrp -Rf proverbot9001 $(REPORT_NAME) $(INDEX_FILES) && \
		  chmod -Rf g+rw $(REPORT_NAME) $(INDEX_FILES) || true'
	mv $(REPORT_NAME) $(REPORT)
	$(MAKE) update-index

publish-weights:
	gzip -k data/pytorch-weights.tar
	rsync -avzP data/pytorch-weights.tar.gz goto:proverbot9001-site/downloads/weights-`date -I`.tar.gz
	ssh goto ln -f proverbot9001-site/downloads/weights-`date -I`.tar.gz proverbot9001-site/downloads/weights-latest.tar.gz

download-weights:
	curl -o data/pytorch-weights.tar.gz proverbot9001.ucsd.edu/downloads/weights-latest.tar.gz
	gzip -d data/pytorch-weights.tar.gz

publish-depv:
	opam info -f name,version menhir ocamlfind ppx_deriving ppx_import cmdliner core_kernel sexplib ppx_sexp_conv camlp5 | awk '{print; print ""}' > known-good-dependency-versions.md

clean:
	rm -rf report-*
	rm -f log*.txt

clean-progress:
	fd '.*\.v\.lin' CompCert | xargs rm -f
	fd '.*\.scrape' CompCert | xargs rm -f
