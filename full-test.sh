#!/usr/bin/env bash
./full-run.sh && ! grep "Failed" scrape.log.txt && ./at-least-n-searched.sh 38
