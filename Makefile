
SHELL=/usr/bin/env bash

ENV_PREFIX=export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$$LD_LIBRARY_PATH

NTHREADS=16
FLAGS=
HIDDEN_SIZE=512

SITE_PATH=goto:/home/alexss/proverbot9001-site

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
	mv data/scrape.txt data/scrape.bkp 2>/dev/null || true
	cd src && \
	cat ../data/compcert-train-files.txt | $(HEAD_CMD) | \
	xargs python3 scrape.py $(FLAGS) -j $(NTHREADS) --output ../data/scrape.txt \
						       	--prelude ../CompCert
report:
	($(ENV_PREFIX) ; cat data/compcert-test-files.txt | $(HEAD_CMD) | \
	xargs ./src/proverbot9001.py report -j $(NTHREADS) --prelude ./CompCert $(FLAGS))

train:
	./src/proverbot9001.py train encdec data/scrape.txt data/pytorch-weights.tar $(FLAGS) --hidden-size $(HIDDEN_SIZE)

publish:
	$(eval REPORT_NAME := $(shell ./reports/get-report-name.py $(REPORT)/))
	mv $(REPORT) $(REPORT_NAME)
	chmod +rx $(REPORT_NAME)
	tar czf report.tar.gz $(REPORT_NAME)
	rsync -avz report.tar.gz $(SITE_PATH)/reports/
	rsync -avz reports/index.js reports/index.css reports/build-index.py $(SITE_PATH)/reports/
	ssh goto 'cd proverbot9001-site/reports && \
                  tar xzf report.tar.gz && \
                  rm report.tar.gz && \
                  ./build-index.py'
	mv $(REPORT_NAME) $(REPORT)

publish-weights:
	gzip -k data/pytorch-weights.tar
	rsync -avzP data/pytorch-weights.tar.gz goto:proverbot9001-site/downloads/weights-`date -I`.tar.gz
	ssh goto ln -f proverbot9001-site/downloads/weights-`date -I`.tar.gz proverbot9001-site/downloads/weights-latest.tar.gz

download-weights:
	curl -o data/pytorch-weights.tar.gz proverbot9001.ucsd.edu/downloads/weights-latest.tar.gz
	gzip -d data/pytorch-weights.tar.gz

publish-depv:
	opam info -f name,version menhir ocamlfind ppx_deriving ppx_import cmdliner core_kernel sexplib ppx_sexp_conv camlp5 > known-good-dependency-versions.md

clean:
	rm -rf report-*
	rm -f log*.txt

clean-lin:
	fd '.*\.v\.lin' CompCert | xargs rm
