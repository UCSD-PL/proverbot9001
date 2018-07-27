
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

.PHONY: scrape report setup

all: scrape report

setup:
	./setup.sh

scrape:
	mv scrape.txt scrape.bkp 2>/dev/null || true
	cat compcert-train-files.txt | $(HEAD_CMD) | \
	xargs python3 scrape.py $(FLAGS) -j $(NTHREADS) --output scrape.txt \
					       --prelude ./CompCert
report:
	($(ENV_PREFIX) ; cat compcert-test-files.txt | $(HEAD_CMD) | \
	xargs ./proverbot9001.py report $(FLAGS) -j $(NTHREADS) --prelude ./CompCert)

train:
	./proverbot9001.py train-encdec scrape.txt pytorch-weights.tar $(FLAGS) --hidden-size $(HIDDEN_SIZE)

publish:
	$(eval REPORT_NAME := $(shell ./reports/get-report-name.py report/))
	mv report $(REPORT_NAME)
	tar czf report.tar.gz $(REPORT_NAME)
	rsync -avz report.tar.gz $(SITE_PATH)/reports/
	rsync -avz reports/index.js reports/index.css reports/build-index.py $(SITE_PATH)/reports/
	ssh goto 'cd proverbot9001-site/reports && \
                  tar xzf report.tar.gz && \
                  rm report.tar.gz && \
                  ./build-index.py'
	mv $(REPORT_NAME) report

publish-weights:
	gzip -k pytorch-weights.tar
	rsync -avzP pytorch-weights.tar.gz goto:proverbot9001-site/downloads/weights-`date -I`.tar.gz
	ssh goto ln -f proverbot9001-site/downloads/weights-`date -I`.tar.gz proverbot9001-site/downloads/weights-latest.tar.gz

download-weights:
	curl -o pytorch-weights.tar.gz proverbot9001.ucsd.edu/downloads/weights-latest.tar.gz
	gzip -d pytorch-weights.tar.gz
