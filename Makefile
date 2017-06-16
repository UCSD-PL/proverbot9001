
NTHREADS=4

.PHONY: scrape report

all: scrape report

scrape:
	cat compcert-scrapable-files.txt | \
	xargs python3 scrape.py -j $(NTHREADS) --output scrape.txt \
					       --prelude ./CompCert
report:
	cat compcert-scrapable-files.txt | \
	xargs python3 scrape.py -j $(NTHREADS) --output scrape.txt \
					       --prelude ./CompCert
