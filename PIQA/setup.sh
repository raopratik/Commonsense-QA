#!/bin/bash


echo "Setting up data"

wget -P Data/ "https://yonatanbisk.com/piqa/data/train.jsonl"
wget -P Data/ "https://yonatanbisk.com/piqa/data/train-labels.lst"
wget -P Data/ "https://yonatanbisk.com/piqa/data/valid.jsonl"
wget -P Data/ "https://yonatanbisk.com/piqa/data/valid-labels.lst"


