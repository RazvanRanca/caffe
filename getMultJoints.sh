#!/bin/bash

fn=${2-"multJoints"}
ls $1 | xargs -i bash -c 'echo -n {}" "; ls -x '$1/'{}' | tr "\t" " " > $fn
