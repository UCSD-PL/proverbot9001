#!/usr/bin/env bash

# exit on error
set -e

TS_SOCKET=/tmp/graphicscard tsp -fn $@
