#!/usr/bin/env bash
#
# Copyright (c) 2025.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Marco Massenzio (marco@alertavert.com)
#

#
# Run this script on the EC2 remote host to setup git

git config --global init.defaultBranch main
git config --global user.name "Marco Massenzio"
git config --global user.email "marco@massenz.io"

# Need to SCP the private key before running this step
mkdir .ssh
read -r -p "scp the private key gh to this host before continuing

Run this command on the local host:
    scp \${HOME}/.ssh/gh ubuntu@cuda-learn:~/.ssh

Hit Enter when done.  "
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/gh

git clone git@github.com:massenz/cuda-learn.git && \
    cd cuda-learn && \
    make all
