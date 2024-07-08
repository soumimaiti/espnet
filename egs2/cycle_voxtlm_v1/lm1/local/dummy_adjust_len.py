import numpy as np
import argparse
import shutil
import os
from tqdm import tqdm


# Adjusts sequence length for generation examples
# Input: stats dir file
# Output: Updated stats dir file
# Call as python dummy_adjust_len.py filename
parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()
print(args.filename)


dir_name = os.path.dirname(args.filename)
base_name = os.path.basename(args.filename)
bk_base_name = os.path.basename(args.filename) + "_bk"
bk_filename = os.path.join(dir_name, bk_base_name)

# copy old file
shutil.copy(args.filename, bk_filename)
print("Copying as: ", bk_filename)

# Update stats with length
n_textgen = 0
n_speechgen = 0

# Change these two lines if generated sequence length changes in code
text_gen_len = 100
speech_gen_len = 300


def update_seq_len(utt_id, utt_seq_len, text_gen_len, speech_gen_len):
    n_textgen = 0
    n_speechgen = 0

    if "speechgen" in utt_id:
        new_utt_len = utt_seq_len + speech_gen_len
        n_speechgen += 1
    elif "textgen" in utt_id:
        new_utt_len = utt_seq_len + text_gen_len
        n_textgen += 1
    else:
        new_utt_len = utt_seq_len

    return new_utt_len, n_textgen, n_speechgen


if "bpe" in args.filename:
    # text_shape.bpe format: utt_id <utt-len>,<bpe-len>
    with open(args.filename, "w") as fw:
        with open(bk_filename) as f:
            for line in tqdm(f):

                split_line = line.strip().split()
                utt_id = split_line[0]
                utt_len_split = split_line[1].split(",")
                utt_seq_len = int(utt_len_split[0])
                utt_bpe_len = int(utt_len_split[1])

                split_line = line.strip().split()

                # utt_id has textlmgen --> add 300
                # utt_id has unitlmgen --> add 100
                new_utt_len, utt_textgen, utt_speechgen = update_seq_len(
                    utt_id, utt_seq_len, text_gen_len, speech_gen_len
                )
                n_textgen += utt_textgen
                n_speechgen += utt_speechgen

                # write to file
                fw.write("{0} {1},{2}\n".format(utt_id, new_utt_len, utt_bpe_len))
else:
    # text_shape format: utt_id <utt-len>
    with open(args.filename, "w") as fw:
        with open(bk_filename) as f:
            for line in tqdm(f):

                split_line = line.strip().split()
                utt_id = split_line[0]
                utt_len = int(split_line[1])

                split_line = line.strip().split()

                # utt_id is in textlm_gen --> add 300
                # utt_id is in unitlm_gen --> add 100
                new_utt_len, utt_textgen, utt_speechgen = update_seq_len(
                    utt_id, utt_seq_len, text_gen_len, speech_gen_len
                )
                n_textgen += utt_textgen
                n_speechgen += utt_speechgen

                # write to file
                fw.write("{0}  {1}\n".format(utt_id, new_utt_len))

print("Updated textgen: ", n_textgen)

print("Updated speechgen: ", n_textgen)
