#!/usr/bin/env bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Modifications Copyright 2019  Nagoya University (author: Takenori Yoshimura)
# Apache 2.0

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/LibriTTS/dev-clean data/dev-clean"
  exit 1
fi

src=$1
dst=$2
wav=$3

#spk_file=$src/../SPEAKERS.txt

mkdir -p $dst || exit 1

#[ ! -d $src ] && echo "$0: no such directory $src" && exit 1
#[ ! -f $spk_file ] && echo "$0: expected file $spk_file to exist" && exit 1


wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
spk2gender=$dst/spk2gender; [[ -f $spk2gender ]] && rm $spk2gender
#TODO: SM: do we know spk2gender?

python3 local/data_prep.py --json ${src} --dst_path ${dst} --wav_path ${wav}

utils/fix_data_dir.sh ${dst}

spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

ntrans=$(wc -l <$trans)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1

utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in $dst"

exit 0
