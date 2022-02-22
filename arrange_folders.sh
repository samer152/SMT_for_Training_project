#!/bin/bash

for f in *baseline_results\*!(*.log*); do
  echo $f
  echo "${f//\\//}"
  if [ "$f" != '*.log' ]
  then
    mkdir -p "${f//\\//}"
  fi
  mv -- "$f"/* "${f//\\//}"
done