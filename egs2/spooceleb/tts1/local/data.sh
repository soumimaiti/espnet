#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=1
trim_all_silence=true

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# if [ -z "${LIBRITTS}" ]; then
#    log "Fill the value of 'LIBRITTS' of db.sh"
#    exit 1
# fi
# db_root=${SPOCELEB}
# data_url=www.openslr.org/resources/60

#SM: harcode db_root 
db_root="/ocean/projects/cis210027p/wzhangn/espnet_spk/egs2/voxceleb/spk1"
# SM: Download not needed
# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#     # log "stage -1: local/donwload_and_untar.sh"
#     # # download the original corpus
#     # if [ ! -e "${db_root}"/LibriTTS/.complete ]; then
#     #     for part in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
#     #         local/download_and_untar.sh "${db_root}" "${data_url}" "${part}"
#     #     done
#     #     touch "${db_root}/LibriTTS/.complete"
#     # else
#     #     log "Already done. Skiped."
#     # fi

#     # download not needed

#     # # download the additional labels
#     # if [ ! -e "${db_root}"/LibriTTS/.lab_complete ]; then
#     #     git clone https://github.com/kan-bayashi/LibriTTSCorpusLabel.git "${db_root}/LibriTTSCorpusLabel"
#     #     cat "${db_root}"/LibriTTSCorpusLabel/lab.tar.gz-* > "${db_root}/LibriTTS/lab.tar.gz"
#     #     cwd=$(pwd)
#     #     cd "${db_root}/LibriTTS"
#     #     for part in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
#     #         gunzip -c lab.tar.gz | tar xvf - "lab/phone/${part}" --strip-components=2
#     #     done
#     #     touch .lab_complete
#     #     rm -rf lab.tar.gz
#     #     cd "${cwd}"
#     # else
#     #     log "Already done. Skiped."
#     # fi
# fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    
    #for name in test; do
    for name in dev test; do
        # Remove silence and re-create wav file
        #python3 local/trim_silence.py --json "${db_root}/voxceleb1_${name}_full_en_filtered.json" --dst_path data/local/${name}

        # Create kaldi data directory with the trimed audio
        local/data_prep.sh "${db_root}/voxceleb1_${name}_full_en_filtered.json" "data/${name}" data/local/${name}

        utils/fix_data_dir.sh "data/${name}"
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: utils/combine_data.sh"
    utils/combine_data.sh data/train data/dev data/test
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
