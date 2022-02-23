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

#echo "delete old folders"
#for f in $(find . -type d -name 'baseline_results*epochs*'); do
#    echo $f
#    rm -r $f
#done

#echo "change log file names"
#cd baseline_results
#for f in $(find . -type d -name '*.log*'); do
#  cd $f
#  echo $f
#  for log in $(find . -type f -name '*.log*'); do
#    echo "mv $log logfile.log"
#    mv "$log" "logfile.log"
#  done
#  cd ../../../
#done
#cd ../

echo "remove unrelevant pth files"
for f in $(find . -type f -name '*2_Compute*.pth'); do
  echo $f
  rm $f
done
for f in $(find . -type f -name '*1_Compute*.pth'); do
  echo $f
  rm $f
done
for f in $(find . -type f -name '*3_Compute*.pth'); do
  echo $f
  rm $f
done
for f in $(find . -type f -name '*4_Compute*.pth'); do
  echo $f
  rm $f
done
