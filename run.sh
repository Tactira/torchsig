#!/bin/bash
exec docker run --rm -u $(id -u ${USER}):$(id -g ${USER}) -e PYTHONPATH=/workspace/code/torchsig -v /scratch:/scratch -v /tmp:/tmp -v `pwd`:/workspace/code/torchsig -ti tactira/torchsig $*

