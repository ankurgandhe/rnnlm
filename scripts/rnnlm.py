import sys 
from ReadConfig import * 
from Corpus import  CreateData,GetVocabAndUNK
from NNLMio import * 
from TrainNNLM    import print_params,write_machine 
from rnn_benchmark_minibatch import * 
import numpy

def convert_to_sparse(x,minibatch=1,N=4096):
    data = zeros((len(x)/minibatch+int(len(x)/minibatch>0),minibatch,N),dtype=theano.config.floatX)
    n = 0
    mb=0
    for i in x:
        if i >=N:
            i=2
        data[n][mb][i] = 1
	mb=mb+1
	if mb >= minibatch:
            n = n+1
	    mb = 0 
    return data
def load_params(fparam):
    pfiles = numpy.load(fparam+"/params.npz")
    params = []
    for p in sorted(pfiles):
	print >> sys.stderr, "loading ",p,"with shape", pfiles[p].shape
	params.append(pfiles[p])
    return params 

def train_nnlm(params):
    ftrain = params['ftrain']
    fdev = params['fdev']
    ftest = params['ftest']
    fvocab = params['fvocab']
    ffreq = params['ffreq']
    ftrainfeat = params['train_feature_file']
    fdevfeat = params['dev_feature_file']
    ftestfeat = params['test_feature_file']
    n_feats =params['n_features']
    ngram = params ['ngram']
    add_unk = params['add_unk']
    use_unk = params['use_unk']
    N_input_layer = params['N']
    P_projection_layer = params['P']
    H_hidden_layer = params['H']
    N_output_layer = params['N']
    learning_rate = params['learning_rate']
    L1_reg= params['L1']
    L2_reg= params['L2']
    n_epochs= params['n_epochs']
    batch_size= params['batch_size']
    adaptive_learning_rate = params['use_adaptive']
    fparam = params['foutparam']
    write_janus = params['write_janus']

    #For RNNLM, ngram = 2 always 
    ngram = 2 

    print >> sys.stderr, "Reading Vocab files", fvocab
    WordID, UNKw,printMapFile = GetVocabAndUNK(fvocab,ffreq,ngram,add_unk,use_unk)
    print >> sys.stderr, 'Reading Training File: ' , ftrain
    TrainData,N_input_layer,N_unk = CreateData(ftrain,WordID,UNKw,ngram,add_unk,use_unk)
    print >> sys.stderr, 'Reading Dev File: ' , fdev
    DevData,N_input_layer,N_unk = CreateData(fdev,WordID,UNKw,ngram,False,use_unk)
    print >> sys.stderr, 'Reading Test File: ' , ftest
    TestData,N_input_layer,N_unk = CreateData(ftest,WordID,UNKw,ngram,False,use_unk)
    if params['write_ngram_files']:
        WriteData(TrainData, ftrain+'.'+str(ngram)+'g')
        WriteData(DevData, fdev+'.'+str(ngram)+'g')
        WriteData(TestData, ftest+'.'+str(ngram)+'g')
        print >> sys.stderr, "ngrams file written... rerun with [ write_ngram_file = False ] for training"
        sys.exit(1)
    if ftrainfeat!="" and fdevfeat!="" and ftestfeat!="":
        print >> sys.stderr, 'Reading training, dev and test Feature Files ', ftrainfeat, fdevfeat, ftestfeat
        NNLMFeatData = load_alldata_from_file(ftrainfeat,fdevfeat,ftestfeat,ngram,n_feats)
    else:
        NNLMFeatData = []
        n_feats = 0
    #Convert data suitable for NNLM training
    NNLMdata = load_alldata(TrainData,DevData,TestData,ngram,N_input_layer)
    foldmodel = params['fmodel']
    if foldmodel.strip()!="":
        OldParams = load_params(foldmodel)
    else:
        OldParams = None
    if not os.path.exists(fparam):
        os.makedirs(fparam)
    print >> sys.stderr,  "Writing system description"
    print_params(foldmodel,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,1)
    write_machine(foldmodel,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,printMapFile,WordID,
fvocab,1)
    print >> sys.stderr, "singletons:", N_unk
    
    #RNNLM 
    minibatch = 1
    ntrain_set_x = NNLMdata[0][0][0]
    ntrain_set_y = NNLMdata[0][1]

    tot_train_size = len(ntrain_set_y)
    data_set_x = convert_to_sparse(ntrain_set_x,N=N_input_layer,minibatch=minibatch)
    sample_size = len(ntrain_set_x)/minibatch+1
    data_set_y = numpy.append(ntrain_set_y,zeros((sample_size*minibatch - tot_train_size,1),dtype=numpy.int32))
    print >> sys.stderr, "Training size:", tot_train_size, ". With batch:", data_set_x.shape
    rnn = MetaRNN(n_in=N_input_layer,n_hidden = H_hidden_layer, n_out = N_input_layer,samples = sample_size,learning_rate = learning_rate,minibatch =minibatch,n_epochs= n_epochs,old_params=OldParams )


    #validation set for keeping track of progress 
    nvalid_set_x = NNLMdata[1][0][0]
    nvalid_set_y = NNLMdata[1][1]
    tot_valid_size = len(nvalid_set_y)
    valid_set_x = convert_to_sparse(nvalid_set_x,N=N_input_layer,minibatch=minibatch)
    valid_sample_size = len(nvalid_set_x)/minibatch+1
    valid_set_y = numpy.append(nvalid_set_y,zeros((valid_sample_size*minibatch - tot_valid_size,1),dtype=numpy.int32))
    print >> sys.stderr, "Validation size:", tot_valid_size, ". With batch:", valid_set_x.shape
   

    final_params = rnn.train_rnn(data_set_x,data_set_y,valid_set_x,valid_set_y)
    fout = fparam+"/params"
    numpy.savez(fout, *final_params)
    print >> sys.stderr, "Parameters written to", fout 

def test_rnnlm(test_data_file,old_param_file, outfile):
    if outfile == "":
        outfile = testfile+".prob"
    ngram,n_feats,N,P,H,number_hidden_layer,WordID = read_machine(old_param_file)
    unkid = -1
    if "<UNK>" in WordID:
        unkid = WordID["<UNK>"]
    

    TestData,N_input_layer,N_unk = CreateData(test_data_file,WordID,[],ngram,False,False)
    NNLMdata = load_data(TestData,ngram,N,1e10,unkid)
    ntest_set_x =  NNLMdata[0][0]
    ntest_set_y =  NNLMdata[1]
    test_set_x = convert_to_sparse(ntest_set_x,N=N,minibatch=1)
    sample_size = len(ntest_set_x)
    OldParams = load_params(old_param_file)
    rnn = MetaRNN(n_in=N,n_hidden = H, n_out = N,samples = sample_size,old_params=OldParams )
    rnn.test_rnn(test_set_x,ntest_set_y,WordID,outfile)

if __name__ == '__main__':
    if len(sys.argv)<2:
        print >> sys.stderr, " usage : python TrainNNLM.py <config.ini> Or test-data param out-file"
        sys.exit(0)
 
    if len(sys.argv)>2:
	print >> sys.stderr, "running test.." 
	test_data_file =  sys.argv[1]
	old_param_file = sys.argv[2]
	out_file = sys.argv[3]
	test_rnnlm(test_data_file,old_param_file, out_file)
	sys.exit(0) 
    configfile = sys.argv[1]
    params = ReadConfigFile(configfile)
    if not params:
        print >> sys.stderr, "Could not read config file... Exiting"
        sys.exit()
    train_nnlm(params)





