#!/bin/bash

# /data/ad6813/pipe-data/blueboxList
# /data/ad6813/pipe-data/mixedList
# /data/ad6813/pipe-data/soil_eval/list
# /data/ad6813/pipe-data/scrape_blue/list

cd $1 
list=${2-"list"}
echo $list
while read line  
do echo "$line"
   jpg=($line)
   eog $jpg".jpg"
done < $list



