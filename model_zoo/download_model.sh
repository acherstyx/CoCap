#!/bin/bash

SCRIPT_DIR=$(dirname -- "$0")
cd "${SCRIPT_DIR}" || exit 1

aria2c -i ./urls.txt -x 8
