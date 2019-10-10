#!/bin/bash
ROOT_PATH=scene_parsing
pip install -r requirements.txt
# Image and model names
TEST_IMG=teaser #ADE_val_00001519.jpg
MODEL_PATH=$ROOT_PATH/ade20k-resnet18dilated-ppm_deepsup #ade20k-mobilenetv2dilated-c1_deepsup 
RESULT_PATH=./

ENCODER=$MODEL_PATH/encoder_epoch_20.pth
DECODER=$MODEL_PATH/decoder_epoch_20.pth

# Download model weights and image
if [ ! -e $MODEL_PATH ]; then
  mkdir $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi

# Inference #ade20k-mobilenetv2dilated-c1_deepsup.yaml
python3 -u app.py \
  --imgs $TEST_IMG \
  --cfg $ROOT_PATH/config/ade20k-resnet18dilated-ppm_deepsup.yaml \
  DIR $MODEL_PATH \
  TEST.result ./result_data \
  TEST.checkpoint epoch_20.pth
