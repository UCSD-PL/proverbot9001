
SHELL=/bin/bash

ENV_PREFIX=export LD_LIBRARY_PATH=$$PWD/darknet/:/usr/local/cuda/lib64/:$$LD_LIBRARY_PATH

NTHREADS=16
REPORT_NAME=$(shell cat <(date +"%Y-%m-%dT%Hd%Md%S%z") <(echo "+") <(git rev-parse HEAD) | tr -d '\n' | tr ':' 'd')
FLAGS=

.PHONY: scrape report setup

all: scrape report

setup:
	./setup.sh

scrape:
	mv scrape.txt scrape.bkp 2>/dev/null || true
ifeq ($(NUM_FILES),)
	cat compcert-train-files.txt | \
	xargs python3 scrape.py $(FLAGS) -j $(NTHREADS) --output scrape.txt \
					       --prelude ./CompCert
else
	cat compcert-train-files.txt | head -n $(NUM_FILES) | \
	xargs python3 scrape.py $(FLAGS) -j $(NTHREADS) --output scrape.txt \
					       --prelude ./CompCert
endif
report:
ifeq ($(NUM_FILES),)
	($(ENV_PREFIX) ; cat compcert-test-files.txt | \
	xargs python3 report.py $(FLAGS) -j $(NTHREADS) --prelude ./CompCert)
else
	($(ENV_PREFIX) ; cat compcert-test-files.txt | head -n $(NUM_FILES) | \
	xargs python3 report.py $(FLAGS) -j $(NTHREADS) --prelude ./CompCert)
endif

train:
	($(ENV_PREFIX) ; ./darknet/darknet rnn train coq.cfg enc.weights -file ./scrape.txt -clear)

publish:
	mv report $(REPORT_NAME)
	tar czf report.tar.gz $(REPORT_NAME)
	rsync -avz report.tar.gz goto:~/proverbot9001-site/reports/
	rsync -avz reports/index.js reports/index.css reports/build-index.py goto:~/proverbot9001-site/reports/
	ssh goto 'cd proverbot9001-site/reports && \
                  tar xzf report.tar.gz && \
                  rm report.tar.gz && \
                  ./build-index.py'
	mv $(REPORT_NAME) report
