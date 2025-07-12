#!/usr/bin/env bash
#
# Run this script on the EC2 remote host to setup git

git config --global init.defaultBranch main
git config --global user.name "Marco Massenzio"
git config --global user.email "marco@massennz.io"

mkdir .ssh

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/gh

# Need to SCP the private key before running this step
read -p "scp the private key gh to this host before continuing"
git clone git@github.com:massenz/cuda-learn.git

cd cuda-learn
make all
