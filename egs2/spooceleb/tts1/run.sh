#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000
n_fft=1024
n_shift=256
win_length=null

#opts="--use_xvector true --xvector_tool rawnet" 
opts="--use_xvector true" # --xvector_tool rawnet" #
# if [ "${fs}" -eq 24000 ]; then
#     # To suppress recreation, specify wav format
#     opts="--audio_format wav "
# else
#     opts="--audio_format flac "
# fi
opts+=" --audio_format wav"
train_set=train
valid_set=dev
test_sets="test"

train_config=conf/train.yaml
inference_config=conf/decode.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
local_data_opts="" #--trim_all_silence true" # trim all silence in the audio

./tts.sh \
    --ngpu 1 \
    --lang en \
    --feats_type raw \
    --local_data_opts "${local_data_opts}" \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --use_xvector true \
    --token_type phn \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
