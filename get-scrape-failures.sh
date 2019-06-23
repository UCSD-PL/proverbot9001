#!/usr/bin/env bash
diff <(find CompCert -name '*.scrape' | sed 's/\.scrape//' | sed 's/CompCert/./' | sort) <(cat data/compcert-train-files.txt |sort) | grep ">" | cut --delimiter=' ' -f 2
