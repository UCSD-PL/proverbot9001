#!/usr/bin/env bash

[ "$#" -eq 1 ] || (echo "1 argument required, $# provided" && exit 1)
md5sum $1 | cut -f1 -d' ' > $1.lin && cat $1 >> $1.lin
