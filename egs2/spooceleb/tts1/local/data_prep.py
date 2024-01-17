#!/usr/bin/env python3

# Copyright 2024 Soumi Maiti
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Trim silence in the audio and re-create wav files."""

import argparse
import fnmatch
import logging
import os
import sys

import numpy as np
import soundfile as sf
import json




def main():
    """Run trimming."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        help="Path of the json file",
    )
    parser.add_argument(
        "--wav_path",
        type=str,
        help="Path of the wav files",
    )
    parser.add_argument(
        "--dst_path",
        type=str,
        help="Path of the destination folder",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # open wav.scp
    f_wav_scp = open(os.path.join(args.dst_path, "wav.scp"), "w")

    # open text
    f_text = open(os.path.join(args.dst_path, "text"), "w")

    # open utt2spk
    f_utt2spk = open(os.path.join(args.dst_path, "utt2spk"), "w")

    # read json file
    with open(args.json) as f_in:
        data_dict = json.load(f_in)

    for key, values in data_dict.items():
        
        audio_path = values['audio_path']
        trans = values['text']

        languge = values['language']
       

        key_base=key.split('(')[0] # key base name before split number
        split_num=key.split('(')[1].split(')')[0] # split number

        uid = key_base.replace("/","-")+"-{:03d}".format(int(split_num))

        # out_audio_path = os.path.join(wav_dir, os.path.basename(audio_path)) 
        out_audio_path = os.path.join(args.wav_path, uid+".wav") 

        f_wav_scp.write("{0} {1}\n".format(uid, out_audio_path))
        
        # how to add trim  part
        #trim_str="sox {} {} trim {} ={}".format(audio_path, out_audio_path, float(start), float(end))
        #f_wav_scp.write("{0} {1} |\n".format(uid, trim_str))

        # write trans
        f_text.write("{0} {1}\n".format(uid, trans))

        # TODO:SM: what is speaker ?

        spk_id=key.split('/')[0]
        
        # write utt2spk
        f_utt2spk.write("{0} {1}\n".format(uid, spk_id))

    f_wav_scp.close()
    f_text.close()
    f_utt2spk.close()
    
    #logging.info(f"Now processing... ({idx + 1}/{len(wavfiles)})")


if __name__ == "__main__":
    main()
