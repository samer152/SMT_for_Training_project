#!/bin/bash

for f in *baseline_results*; do
  echo $f
  echo "${f//\\//}"
  mv -- "$f" "${f//\\//}"
done