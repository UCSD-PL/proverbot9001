
SHELL=/bin/bash

NTHREADS=4
NUM_FILES=`wc -l < compcert-scrapable-files.txt`
REPORT_NAME=$(shell cat <(date -Iseconds) <(git rev-parse HEAD) | tr -d '\n' | tr ':' 'd')

.PHONY: scrape report setup

all: scrape report

setup:
	./setup.sh

scrape:
	cat compcert-scrapable-files.txt | head -n $(NUM_FILES) | \
	xargs python3 scrape.py -j $(NTHREADS) --output scrape.txt \
					       --prelude ./CompCert
report:
	cat compcert-scrapable-files.txt | head -n $(NUM_FILES) | \
	xargs python3 report.py -j $(NTHREADS) --prelude ./CompCert

publish:
	mv report $(REPORT_NAME)
	tar czf report.tar.gz $(REPORT_NAME)
	rsync -avz report.tar.gz goto:~/proverbot9001-site/reports/
	ssh goto 'cd proverbot9001-site/reports && \
                  tar xzf report.tar.gz && \
                  rm report.tar.gz && \
                  ./build-index.sh'
