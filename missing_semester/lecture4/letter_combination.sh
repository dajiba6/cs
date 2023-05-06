#!/bin/bash

# Loop through all lowercase letters
for i in {a..z}; do
  # Loop through all lowercase letters again
  for j in {a..z}; do
    # Concatenate the two letters and write to file
    echo "${i}${j}" >> 2_letter_combinations.txt
  done
done
