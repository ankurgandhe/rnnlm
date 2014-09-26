if [ $# -lt 2 ]; then
        echo "Usage: sh test.sh gpu-id config-file"
        exit
fi

export PYTHONPATH=~/tools/NNLM/scripts 

gpu=$1
config=$2 
THEANO_FLAGS='base_compiledir=/var/tmp/ankurgan/'$gpu,'device='$gpu python scripts/rnnlm.py $config 
