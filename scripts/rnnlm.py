import sys 
sys.path.append('/home/ankurgan/tools/NNLM/scripts')
from ReadConfig import * 
from Corpus import  CreateData,GetVocabAndUNK
from NNLMio import * 
from TrainNNLM    import print_params,write_machine 
from rnn_vanilla_minibatch import * # benchmark_minibatch import * 
from rnn_gpu import * 
import numpy

def convert_to_sparse_data(x,minibatch=1,N=4096,unk_id = 2,n_feats =0, ntrain_set_x_feat= [] ):
    #data = zeros((len(x)+int(minibatch-len(x)%minibatch),N),dtype=theano.config.floatX)
    data = zeros((len(x),N+n_feats),dtype=theano.config.floatX)
    n = 0
    count_unk = 0 
    for i in x:
        if i >=N:
            i=unk_id 
	if i == unk_id:
	    n = n + 1 
	    count_unk = count_unk + 1 
	    continue 
        data[n][i] = 1
	if n_feats > 0:
	    feature_id = ntrain_set_x_feat[n]
	    data[n][feature_id+N]=1
        n = n+1
    print >> sys.stderr, "Number of unkowns:", count_unk 
    print >> sys.stderr, "Number of ones:",numpy.sum(data)
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
    fout = fparam+"/params"
    write_janus = params['write_janus']
    gpu_copy_size = params['copy_size']
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
    print_params(foldmodel,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, 
   		L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,1)
    write_machine(foldmodel,ngram,N_input_layer,n_feats,P_projection_layer,H_hidden_layer,learning_rate, 
		L1_reg, L2_reg, n_epochs,batch_size,adaptive_learning_rate,fparam,printMapFile,WordID,fvocab,1)
    
    #RNNLM 
    minibatch = batch_size
    unk_id = -1
    if "<UNK>" in WordID:
        unk_id = WordID["<UNK>"]

    # Gather training data 
    ntrain_set_x = NNLMdata[0][0][0]
    ntrain_set_y = NNLMdata[0][1]
    if n_feats > 0:
	ntrain_set_x_feat = NNLMFeatData[0][0][0]
    else:
	ntrain_set_x_feat = [] 

    tot_train_size = len(ntrain_set_y)
    data_set_x = convert_to_sparse_data(ntrain_set_x,N=N_input_layer,minibatch=minibatch,
		 			unk_id = unk_id,n_feats=n_feats,ntrain_set_x_feat=ntrain_set_x_feat)
    sample_size = len(ntrain_set_x)
    data_set_y = numpy.append(ntrain_set_y,zeros(( data_set_x.shape[0] - tot_train_size,1),dtype=numpy.int32))
    print >> sys.stderr, "Training size:", data_set_x.shape[0], ". With batch size:", minibatch 


    #validation set for keeping track of progress 
    nvalid_set_x = NNLMdata[1][0][0]
    nvalid_set_y = NNLMdata[1][1]
    tot_valid_size = len(nvalid_set_y)
    valid_set_x = convert_to_sparse_data(nvalid_set_x,N=N_input_layer,minibatch=minibatch,unk_id = unk_id,
					 n_feats=n_feats,ntrain_set_x_feat=ntrain_set_x_feat)
    valid_sample_size = len(nvalid_set_x) 
    valid_set_y = numpy.append(nvalid_set_y,zeros((valid_set_x.shape[0] - tot_valid_size,1),dtype=numpy.int32))
    print >> sys.stderr, "Validation size:", valid_set_x.shape[0] , ". With batch size:", minibatch

    #rnn = MetaRNN(n_in=N_input_layer,n_hidden = H_hidden_layer,  n_features = n_feats, n_out = N_input_layer,samples = sample_size,
    #             learning_rate = learning_rate,minibatch =minibatch,n_epochs= n_epochs,old_params=OldParams )
    rnn = GpuRNN(n_in=N_input_layer,n_hidden = H_hidden_layer, n_features = n_feats, n_out = N_input_layer,samples = sample_size,
                 learning_rate = learning_rate,minibatch =minibatch,n_epochs= n_epochs,old_params=OldParams )

   
    final_params = rnn.train_rnn(data_set_x,data_set_y,valid_set_x,valid_set_y,gpu_copy_size,unk_id,fout)
    numpy.savez(fout, *final_params)
    print >> sys.stderr, "Learnt Parameters written to", fout 

def test_rnnlm(test_data_file, test_data_feature_file, old_param_file, outfile):
    if outfile == "":
        outfile = testfile+".prob"
    ngram,n_feats,N,P,H,number_hidden_layer,WordID = read_machine(old_param_file)
    minibatch = 2
    unk_id = -1
    if "<UNK>" in WordID:
        unk_id = WordID["<UNK>"]
    

    TestData,N_input_layer,N_unk = CreateData(test_data_file,WordID,[],ngram,False,False)
    NNLMdata = load_data(TestData,ngram,N,1e10,unk_id)
    ntest_set_x =  NNLMdata[0][0]
    ntest_set_y =  NNLMdata[1]
    if n_feats > 0:
	print >> sys.stderr, "reading features from ", test_data_feature_file
        NNLMFeatData = load_data_from_file(test_data_feature_file,ngram,N,200000,unk_id)
        ntrain_set_x_feat = NNLMFeatData[0][0]
    else:
        ntrain_set_x_feat = []

    test_set_x = convert_to_sparse_data(ntest_set_x,N=N,minibatch=0,unk_id=unk_id,
					n_feats=n_feats,ntrain_set_x_feat=ntrain_set_x_feat )

    test_set_y = numpy.append(ntest_set_y,zeros((test_set_x.shape[0] - len(ntest_set_y),1),dtype=numpy.int32))
    sample_size = len(ntest_set_x)
    OldParams = load_params(old_param_file)
    
    rnn = GpuRNN(n_in=N,n_hidden = H,  n_features = n_feats, n_out = N,samples = sample_size,old_params=OldParams )
    rnn.test_rnn_batch(test_set_x,test_set_y,WordID,outfile)

if __name__ == '__main__':
    if len(sys.argv)<2:
        print >> sys.stderr, " usage : python TrainNNLM.py <config.ini> Or test-data param out-file"
        sys.exit(0)
 
    if len(sys.argv)>2:
	print >> sys.stderr, "running test.." 
	test_data_file =  sys.argv[1]
	test_data_feature_file = sys.argv[2]
	old_param_file = sys.argv[3]
	out_file = sys.argv[4]
	test_rnnlm(test_data_file,test_data_feature_file,old_param_file, out_file)
	sys.exit(0) 
    configfile = sys.argv[1]
    params = ReadConfigFile(configfile)
    if not params:
        print >> sys.stderr, "Could not read config file... Exiting"
        sys.exit()
    train_nnlm(params)





