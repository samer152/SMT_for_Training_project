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
for f in *baseline_results*; do
  if [ "$f" != '*.log' ]
  then
    echo $f
  fi
done

echo "change log file names"
for f in $(find . -type f -name '*.log*'); do
  echo $f
done
