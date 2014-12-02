#!/bin/bash

cd $1 
list=${2-"list"}
while read line  
do echo "$line"
   jpg=($line)
   eog $jpg".jpg"
done < $list
