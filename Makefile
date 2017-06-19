
NTHREADS=4
REPORT_SIZE=`wc -l < compcert-scrapable-files.txt`

.PHONY: scrape report setup

all: scrape report

setup:
	./setup.sh

scrape:
	cat compcert-scrapable-files.txt | \
	xargs python3 scrape.py -j $(NTHREADS) --output scrape.txt \
					       --prelude ./CompCert
report:
	cat compcert-scrapable-files.txt | head -n $(REPORT_SIZE) | \
	xargs python3 report.py -j $(NTHREADS) --prelude ./CompCert
