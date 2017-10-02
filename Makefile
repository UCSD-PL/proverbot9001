
SHELL=/bin/bash

ENV_PREFIX=export LD_LIBRARY_PATH=$$PWD/darknet/:/usr/local/cuda/lib64/:$$LD_LIBRARY_PATH

NTHREADS=16
FLAGS=
HIDDEN_SIZE=512

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
	xargs python3 report.py $(FLAGS) -j $(NTHREADS) --prelude ./CompCert)

train:
	./predict_tactic.py --train --save pytorch-weights --hiddensize $(HIDDEN_SIZE)

publish:
	$(eval REPORT_NAME := $(shell ./reports/get-report-name.py report/))
	mv report $(REPORT_NAME)
	tar czf report.tar.gz $(REPORT_NAME)
	rsync -avz report.tar.gz goto:~/proverbot9001-site/reports/
	rsync -avz reports/index.js reports/index.css reports/build-index.py goto:~/proverbot9001-site/reports/
	ssh goto 'cd proverbot9001-site/reports && \
                  tar xzf report.tar.gz && \
                  rm report.tar.gz && \
                  ./build-index.py'
	mv $(REPORT_NAME) report
