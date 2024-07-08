dir=$1

# dir: stats dir , example: exp/lm_stats_en_bpe10000lm
# change following files
# dir/ 
#    train
#		text_shape  text_shape.bpe
#	valid
#		text_shape  text_shape.bpe
#

# for train valid
for sub_dir in "train" "valid"; do
    sub_dir_path=${dir}/${sub_dir}
    echo "${sub_dir}"
    for file in "text_shape.bpe" "text_shape"; do
        echo "${sub_dir_path}/${file}"
        python local/dummy_adjust_len.py ${sub_dir_path}/${file}
    done
done