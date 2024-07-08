import numpy as np
import os
from tqdm import tqdm
import argparse


# given dump train/valid file craetes task_id file in the dump_dir
# creates a file with task ids
# 2 : text generation tasks, lines ends with <generatetext>
# 1 : speech generation tasks, lines ends with <generatespeech>
# 0 : default
# Call as python create_task_file.py dump/raw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dump_dir")
    args = parser.parse_args()
    print(args.dump_dir)

    dump_dir = args.dump_dir

    # train
    with open(os.path.join(dump_dir, "train/text")) as f:
        with open(os.path.join(dump_dir, "train/text_taskid"), "w") as fw:
            for line in tqdm(f):
                line = line.strip()
                uid = line.split()[0]
                if line.endswith("<generatetext>"):
                    fw.write("{0}  2\n".format(uid))
                elif line.endswith("<generatespeech>"):
                    fw.write("{0}  1\n".format(uid))
                else:
                    fw.write("{0}  0\n".format(uid))
    # valid
    with open(os.path.join(dump_dir, "valid/text")) as f:
        with open(os.path.join(dump_dir, "valid/text_taskid"), "w") as fw:
            for line in tqdm(f):
                line = line.strip()
                uid = line.split()[0]
                if line.endswith("<generatetext>"):
                    fw.write("{0}  2\n".format(uid))
                elif line.endswith("<generatespeech>"):
                    fw.write("{0}  1\n".format(uid))
                else:
                    fw.write("{0}  0\n".format(uid))
