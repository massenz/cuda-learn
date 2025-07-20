#!/usr/bin/env bash
#
# Copyright (c) 2025 AlertAvert.com.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Marco Massenzio (marco@alertavert.com)
#

git config --global init.defaultBranch main
git config --global user.name "Marco Massenzio"
git config --global user.email "marco@massenz.io"

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/gh.pem

git clone git@github.com:massenz/cuda-learn.git && \
    cd cuda-learn && \
    make all
