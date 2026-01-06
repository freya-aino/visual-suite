#!/bin/bash

touch dummy_file

# Archive the model
torch-model-archiver \
    --model-name edgeface \
    --version 1.0 \
    --handler handler.py \
    --extra-files edgeface

torchserve --start --foreground \
    --model-store . \
    --models ./edgeface.mar \
    --ts-config ./config.properties