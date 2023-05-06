#!/usr/bin/env bash

count=0
while true; do
	((count++))
	echo "running iteration $count times"
	if ./task3.sh >> output.txt 2>> error.txt;then
		echo "success!"
	else
		echo "failed!"
		break
	fi

done

echo "==output=="
cat output.txt
echo "==error=="
cat error.txt
echo "iterated $count times"
