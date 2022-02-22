#!/bin/bash

for f in *baseline_results*; do
  echo $f
  echo "${f//\\//}"
  if [ "$f" != '*.log' ]
  then
    mkdir -p "${f//\\//}"
  fi
  mv -- "$f"/* "${f//\\//}"
done

for f in *baseline_results*.log*; do
  echo $f
  mv -- "$f" "${f//\\//}"
done