#tokenize data
python ../../scripts/spm_encode.py \
  --model ../../../yo_en/m2m_model/spm.128k.model \
  --output_format=piece \
  --inputs=data/en_train.txt \
  --outputs=processed_data/train.en

python ../../scripts/spm_encode.py \
  --model ../../../yo_en/m2m_model/spm.128k.model \
  --output_format=piece \
  --inputs=data/es_train.txt \
  --outputs=processed_data/train.es

python ../../scripts/spm_encode.py \
  --model ../../../yo_en/m2m_model/spm.128k.model \
  --output_format=piece \
  --inputs=data/en_val.txt \
  --outputs=processed_data/val.en

python ../../scripts/spm_encode.py \
  --model ../../../yo_en/m2m_model/spm.128k.model \
  --output_format=piece \
  --inputs=data/es_val.txt \
  --outputs=processed_data/val.es

#binarize data
fairseq-preprocess \
    --source-lang en --target-lang es \
    --trainpref train \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir processed_data \
    --srcdict ../../../yo_en/m2m_model/model_dict.128k.txt --tgtdict ../../../yo_en/m2m_model/model_dict.128k.txt

