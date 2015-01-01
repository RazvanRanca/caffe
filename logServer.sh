#!/bin/bash

server=${1:-"07"}
if [ "$server" == "root" ]
then
  ssh -X root@78.129.148.77 
else
  sshpass -p 9northerncighT! ssh -A -t ad6813@shell2.doc.ic.ac.uk 'ssh -A -t graphic'$server'.doc.ic.ac.uk "cd /data/ad6813/pipe-data/Bluebox; bash -l"'
fi
