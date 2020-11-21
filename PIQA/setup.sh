#!/bin/bash


echo "Fetching PIQA Dataset"

wget -P Data/ "https://yonatanbisk.com/piqa/data/train.jsonl"
wget -P Data/ "https://yonatanbisk.com/piqa/data/train-labels.lst"
wget -P Data/ "https://yonatanbisk.com/piqa/data/valid.jsonl"
wget -P Data/ "https://yonatanbisk.com/piqa/data/valid-labels.lst"


