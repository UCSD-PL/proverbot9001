
SHELL=/bin/bash

NTHREADS=4
REPORT_NAME=$(shell cat <(date -Iseconds) <(echo "+") <(git rev-parse HEAD) | tr -d '\n' | tr ':' 'd')
FLAGS=

.PHONY: scrape report setup

all: scrape report

setup:
	./setup.sh

scrape:
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
ifeq($(NUM_FILES),)
	cat compcert-test-files.txt | \
	xargs python3 report.py $(FLAGS) -j $(NTHREADS) --prelude ./CompCert
else
	cat compcert-test-files.txt | head -n $(NUM_FILES) | \
	xargs python3 report.py $(FLAGS) -j $(NTHREADS) --prelude ./CompCert
endif

publish:
	mv report $(REPORT_NAME)
	tar czf report.tar.gz $(REPORT_NAME)
	rsync -avz report.tar.gz goto:~/proverbot9001-site/reports/
	ssh goto 'cd proverbot9001-site/reports && \
                  tar xzf report.tar.gz && \
                  rm report.tar.gz && \
                  ./build-index.sh'
	mv $(REPORT_NAME) report
