#!/bin/bash
source /home/hanoch/.virtualenvs/tir_od/bin/activate
if [ -z $1 ] ; then
  python -u /home/hanoch/projects/tir_od/yolov7/tools/merge_results.py
else
  python -u /home/hanoch/projects/tir_od/yolov7/tools/merge_results.py --path "$1"
fi