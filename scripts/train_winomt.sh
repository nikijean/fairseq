#!/bin/bash
set -e
\rm -rf /checkpoint/adirendu/spare_embeddings_training/fine-tune-checkpoint/
export MKL_THREADING_LAYER=GNU
CUDA_VISIBLE_DEVICES=1 fairseq-train winomt_data_bin_expanded \
  --finetune-from-model  /checkpoint/adirendu/spare_embeddings_training/pre-trained/418M_last_checkpoint.pt \
  --save-dir /checkpoint/adirendu/spare_embeddings_training/winomt/fine-tune-checkpoint/ \
  --task translation_multi_simple_epoch \
  --encoder-normalize-before \
  --lang-pairs 'en-es' \
  --batch-size 10 \
  --decoder-normalize-before \
  --encoder-langtok src \
  --decoder-langtok \
  --criterion cross_entropy \
  --optimizer adafactor \
  --lr-scheduler cosine \
  --lr 3e-05 \
  --max-update 40000 \
  --update-freq 2 \
  --save-interval 1 \
  --save-interval-updates 5000 \
  --keep-interval-updates 10 \
  --no-epoch-checkpoints \
  --log-format simple \
  --log-interval 2 \
  --patience 10 \
  --arch transformer_wmt_en_de_big \
  --encoder-layers 12 --decoder-layers 12 \
  --share-decoder-input-output-embed \
  --ddp-backend no_c10d \
  --max-epoch 10 \
  --split-embeddings 2
