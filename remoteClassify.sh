#!/bin/bash

sshpass -p 9northerncighT! ssh -A -t ad6813@shell2.doc.ic.ac.uk "ssh -A -t graphic06.doc.ic.ac.uk 'bash -c \"source ~/.bashrc ; python classifyPipe.py --gpu --pretrained_model task/misal/_iter_1200.caffemodel /data/ad6813/pipe-data/Bluebox/100125.jpg temp\"'"
