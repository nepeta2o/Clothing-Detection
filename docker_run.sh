#!/bin/sh
set -e
docker build -t clothing-detection .
docker run -it --rm -p 5002:5002 clothing-detection
