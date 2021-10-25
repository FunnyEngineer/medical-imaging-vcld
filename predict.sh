#!/bin/bash
wget https://www.dropbox.com/s/tow4zxu6r5h52i4/best_model?dl=1 -O ./src/best_model
python3 src/predict.py $1 $2
