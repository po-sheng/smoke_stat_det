#!/bin/bash

# find all file under given directory
dirPath="../data/train/"
files=`ls $dirPath`

# iteratively rm the whitespace to "_" in file name 
IFS=$'\n'
for file in $files; do
    echo "$file"
    newFile=`echo "$file" | sed 's/ /_/g'`
    echo "$newFile"
    if [ "$newFile" != "$file" ]; then
        mv "$dirPath$file" "$dirPath$newFile"
    fi
done
