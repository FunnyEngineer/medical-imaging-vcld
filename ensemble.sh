#!/bin/bash
wget https://www.dropbox.com/s/d2ahuf3mbsq02gc/models.zip?dl=1 -O models.zip
unzip models.zip
python3 src/ensemble.py $1 $2 ./output_csv ./models
