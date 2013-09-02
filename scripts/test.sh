if [ $# -lt 3 ]; then
        echo "Usage: sh test.sh test-data model-dir out-file"
        exit
fi

export PYTHONPATH=~/tools/NNLM/scripts 
test_data_file=$1
old_param_file=$2 
out_file=$3
THEANO_FLAGS='device=gpu1' python scripts/rnnlm.py $1 $2 $3 
