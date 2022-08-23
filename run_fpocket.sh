#!/bin/bash
docker run --user root -u $(id -u):$(id -g) -v /:/WORKDIR fpocket/fpocket fpocket -f /WORKDIR/$1
