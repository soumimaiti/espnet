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
import tqdm


def main():
    """Run trimming."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        help="Path of the json file",
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

    # read json file
    with open(args.json) as f_in:
        data_dict = json.load(f_in)

    for key, values in tqdm.tqdm(data_dict.items()):
        
        audio_path = values['audio_path']
        start = float(values['start'])
        end = float(values['end'])

        key_base=key.split('(')[0] # key base name before split number
        split_num=key.split('(')[1].split(')')[0] # split number
        uid = key_base.replace("/","-")+"-{:03d}".format(int(split_num))

        x, fs = sf.read(audio_path)
        start_idx = int(start * fs)
        end_idx = int(end * fs)

        new_x = x[start_idx:end_idx]
        
        write_path = os.path.join(args.dst_path, uid+".wav") 
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        sf.write(write_path, new_x, fs)




if __name__ == "__main__":
    main()
