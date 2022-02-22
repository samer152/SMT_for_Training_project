#!/bin/bash

#echo "make folders"
#for f in *baseline_results*; do
#  echo $f
#  echo "${f//\\//}"
#  if [ "$f" != '*.log' ]
#  then
#    mkdir -p "${f//\\//}"
#  fi
#  mv -- "$f"/* "${f//\\//}"
#done
#
#echo "move log files"
#for f in *baseline_results*.log*; do
#  echo $f
#  mv -- "$f" "${f//\\//}"
#done

echo "delete old folders"
for f in $(find . -type d -name 'baseline_results*epochs*'); do
    echo $f
done

echo "change log file names"
cd baseline_results
for f in $(find . -type f -name '*.log*'); do
  echo $f
done
cd ../
