#!/bin/bash

filename=$(basename "$1")
echo "Processing "$filename
if ! ssh ad6813@shell2.doc.ic.ac.uk "test -e '/homes/ad6813/CP_Demo/$filename'"
then echo "Copying to server"; scp -r $1 ad6813@shell2.doc.ic.ac.uk:/homes/ad6813/CP_Demo/
fi
sshpass -p 9northerncighT! ssh -A -t ad6813@shell2.doc.ic.ac.uk "ssh -A -t graphic06.doc.ic.ac.uk 'bash -c \"source ~/.bashrc ; python classifyPipe.py --gpu --pretrained_model task/misal/_iter_1200.caffemodel /homes/ad6813/CP_Demo/$filename temp\"'"
