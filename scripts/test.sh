if [ $# -lt 5 ]; then
        echo "Usage: sh test.sh gpu-id test-data test-data-feat model-dir out-file"
        exit
fi

export PYTHONPATH=~/tools/NNLM/scripts
gpu=$1 
test_data_file=$2
test_data_feature_file=$3
old_param_file=$4
out_file=$5
THEANO_FLAGS='base_compiledir=/var/tmp/ankurgan/'$gpu,'device='$gpu python scripts/rnnlm.py $test_data_file $test_data_feature_file $old_param_file $out_file
