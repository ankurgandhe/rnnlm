export PYTHONPATH=~/tools/NNLM/scripts 

THEANO_FLAGS='device=gpu0' python scripts/rnnlm.py $1
