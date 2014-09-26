"""
    This file contains a RNN benchmark for a standard RNN with tap -1 and minibatches.
    It uses a softmax output layer. It can be used to compare THEANO with another
    RNN code snippet.
    
    This version must have equal sequence lengths (padded). 
    However it's faster than the normal version
        
    data format:
        
        input  ...  tensor3():[N][seq_length][frame]
        output ...  vector:[target1|...|targetN]
    
        access a input sequence N via input[N]. To access a special
        frame N in sequence N simply type input[N][N][:]
        
        access a target (output) N via the indexTable idx
            target[idx['target'][N]:idx['target'][N+1]]
    
    NOTE:    
        - Please take care that you only compare equal networks with equal datasets.
        - taps greater [-1] are not supported yet (although there are the for loops in step),
          due to tensor4() inconsistency
    BUGS:
        - there are some bugs with shared datasets, which have to be fixed (see line:206)
"""

import time, sys
import numpy
import theano
import theano.tensor as T
from theano import sandbox 
import math 
from rnn_vanilla_minibatch import * 


#---------------------------------------------------------------------------------
useWf=1

class GpuRNN(object):

    def __init__(self,n_in=784,n_hidden=100,  n_features = 0, n_out=11,samples=1000,
                learning_rate=0.1,minibatch=5,
                n_epochs=1,old_params = None,
                output_taps=[-1]):

        #-----------------------------------------
        # THEANO SETUP
        #-----------------------------------------
        # setup mode
        self.mode = theano.Mode(linker='cvm')
        # setup profile
        self.profile = 0

        #-----------------------------------------
        # MODEL SETUP
        #-----------------------------------------
	self.n_batches = samples/minibatch ; 
        self.N = minibatch #samples/n_batches  # number of samples
        self.n_in = n_in # number of input units
        self.n_hidden = n_hidden # number of hidden units
        self.n_out = n_out # number of output units
	self.n_features = n_features 
        self.minibatch = minibatch # sequence length
        self.lr = learning_rate 
        self.n_epochs  = n_epochs
	self.output_taps = output_taps 
        self.old_params = old_params
	self.samples = samples 
	self.test_minibatch = self.minibatch*100
        self.hidden_init = theano.shared(numpy.zeros((n_hidden,)).astype(theano.config.floatX), name='h') #numpy.zeros((n_hidden,),dtype=theano.config.floatX)
	if useWf:
	    print >> sys.stderr, 'network: n_in:{},n_features:{},n_hidden:{},n_out:{},output:softmax'.format(n_in,n_features, n_hidden, n_out) 
	else:
            print >> sys.stderr, 'network: n_in:{},n_hidden:{},n_out:{},output:softmax'.format(n_in+n_features, n_hidden, n_out)
        print >> sys.stderr, 'data: samples:{},batch_size:{}'.format(samples,self.minibatch)
        
        #-----------------------------------------
        # RNN SETUP
        #-----------------------------------------
        # initialize random generator
        self.rng = numpy.random.RandomState(1234)
        # construct the CTC_RNN class
	if useWf:
            self.classifier = RNN(rng=self.rng, output_taps=output_taps, n_in=n_in, n_hidden=n_hidden, n_out=n_out, 
			      samples=self.N, mode=self.mode, profile=self.profile,params=old_params,n_features=n_features)
	else:
            self.classifier = RNN(rng=self.rng, output_taps=output_taps, n_in=n_in+n_features, n_hidden=n_hidden, n_out=n_out,
                              samples=self.N, mode=self.mode, profile=self.profile,params=old_params)


    def train_rnn(self,data_x,data_y,valid_x=None,valid_y=None,gpu_copy_size=10000,unk_id=2,fout=""):
        #--------------------------------------------
        # Shape of X x= Nx1xn_in , shape of y : N
        #--------------------------------------------

	if valid_x == None or valid_y == None:
	    print >> sys.stderr, "No validation set provided. Running till set number of epochs" 
	    test_fn = None 
	else:
	    test_sample = valid_x.shape[0]
	    test_batches = test_sample / self.test_minibatch 
	    if useWf:
	    	test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in, 
				  n_hidden=self.n_hidden, n_out=self.n_out, 
				  samples=self.test_minibatch, mode=self.mode, profile=self.profile,params = self.old_params,n_features = self.n_features)
	    else:
                test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in+self.n_features,
                                  n_hidden=self.n_hidden, n_out=self.n_out,
                                  samples=self.test_minibatch, mode=self.mode, profile=self.profile,params = self.old_params)

	    test_fn = test_classifier.build_test_function(self.mode, self.profile)

	get_params = self.classifier.build_get_params(self.mode,self.profile);	
        best_validation_cost = 1e10 ;

        print >> sys.stderr, 'Running ({} epochs), each with {} batches'.format(self.n_epochs,self.n_batches)
        start_time = time.clock()
  	n_gpu_batches = self.samples / gpu_copy_size +1 ; 
	print >> sys.stderr, "data will be copied to gpu in", n_gpu_batches, "batches"
        for iepoch in xrange(self.n_epochs) :
	    total_cost = 0
            self.hidden_init = numpy.zeros((self.n_hidden,),dtype=theano.config.floatX)
            self.lr = self.lr / 1.5
            if self.lr < 0.025*2 :
                self.lr = 0.025*2
            print >> sys.stderr, "Epoch:",iepoch,", alpha:",self.lr
	    ncount = 0
	    for gbatch in xrange(n_gpu_batches):
		if n_gpu_batches>1 or iepoch==0:
	    	    data_x_shared = theano.shared(data_x[gbatch*gpu_copy_size:min(self.samples,(gbatch+1)*gpu_copy_size)])
	    	    data_y_shared = T.cast(theano.shared(data_y[gbatch*gpu_copy_size:min(self.samples,(gbatch+1)*gpu_copy_size)]),'int32')
            	    self.train_fn = self.classifier.build_finetune_functions_gpu( data_x_shared, data_y_shared, 
									      self.lr ,self.mode, self.profile)

		n_batches = len(data_y[gbatch*gpu_copy_size:min(self.samples,(gbatch+1)*gpu_copy_size)])/self.N 
                print >> sys.stderr,"shuffling data... ",
                xn = numpy.arange(n_batches)
                numpy.random.shuffle(xn)
                print >> sys.stderr,"shuffled"
                for ibatch in xn:
            	    #for ibatch in range(n_batches):
                    batch_cost,final_hidden  = self.train_fn(ibatch) #,self.hidden_init) #data_x1, data_y1,self.hidden_init)
                    total_cost = total_cost + batch_cost 
		    ncount = ncount + 1
                    if (ibatch%5000)==1:
                    	print >> sys.stderr , "Current Train entropy:", total_cost/numpy.log10(2)/((ncount))
                print >> sys.stderr , "Current Train entropy:", total_cost/numpy.log10(2)/((ncount))

		if n_gpu_batches >1:
           	    del data_x_shared
		    del data_y_shared 
	    self.learnt_params = get_params()
	    self.hidden_init = numpy.zeros((self.n_hidden,),dtype=theano.config.floatX)
	    if useWf:
            	test_classifier.initialize(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in, n_hidden=self.n_hidden, 
				       n_out=self.n_out, samples=self.test_minibatch, mode=self.mode, profile=self.profile,params = self.learnt_params,n_features=self.n_features)
	    else:
                test_classifier.initialize(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in+self.n_features, n_hidden=self.n_hidden,
                                       n_out=self.n_out, samples=self.test_minibatch, mode=self.mode, profile=self.profile,params = self.learnt_params)

	    if test_fn != None:
	        test_fn = test_classifier.build_test_function(self.mode, self.profile)
		validation_cost = 0 
		n_words = 0 
		unk_words = 0 
		for ibatch in xrange(test_batches):
	    	    probs,last_hidden = test_fn(valid_x[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch] ,
						valid_y[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch],self.hidden_init)
		    self.hidden_init = last_hidden
		   
		    for y,logp in zip(valid_y[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch],probs):
			if y!= unk_id:
			    validation_cost = validation_cost + logp 
			    n_words = n_words + 1 
		    	else:
			    unk_words = unk_words + 1 
	    else:
	    	probs = 0 
		validation_cost = 0 

	    validation_cost = -1 * validation_cost / numpy.log10(2) / n_words 
	    print >> sys.stderr, "\tNumber of unks:",unk_words, "Validation Entropy: ", validation_cost 
	    if best_validation_cost >= validation_cost :
	    	best_validation_cost = validation_cost 
        	self.final_params = get_params()
		if fout!="":
	            numpy.savez(fout, *self.final_params)
	print >> sys.stderr, "Best Validation entropy:", best_validation_cost 
	return self.final_params 
	
    def test_rnn(self,test_x,test_y,WordID,outfile):
	test_sample = test_x.shape[0]
	self.hidden_init = numpy.zeros((self.n_hidden,),dtype=theano.config.floatX)
	if useWf:
	    test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in, n_hidden=self.n_hidden, 
			      n_out=self.n_out, samples=test_sample, mode=self.mode, profile=self.profile,params = self.old_params,n_features=self.n_features)
	else:
            test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in+self.n_features, n_hidden=self.n_hidden,
                              n_out=self.n_out, samples=test_sample, mode=self.mode, profile=self.profile,params = self.old_params)

	test_fn = test_classifier.build_test_function(self.mode, self.profile)
	probs,last_hidden = test_fn(test_x,test_y,self.hidden_init) 
	
        test_cost  = -1*numpy.mean(probs) 
	print >> sys.stderr , "Total ppl: ", test_cost 
	fout = open(outfile,'w')
	for y,yl in zip(test_y,probs):
	    if y==WordID['<UNK>']:
		print >> fout, 0
                continue
            print >> fout,math.pow(10,yl) #numpy.exp(yl)

    def test_rnn_batch(self,test_x,test_y,WordID,outfile):
        test_sample = test_x.shape[0]
	self.hidden_init = numpy.zeros((self.n_hidden,),dtype=theano.config.floatX)
	if useWf:
            test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in, n_hidden=self.n_hidden, 
				   n_out=self.n_out, samples=self.test_minibatch, mode=self.mode, profile=self.profile,params = self.old_params,n_features=self.n_features)
	else:
            test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in+self.n_features, n_hidden=self.n_hidden,
                                   n_out=self.n_out, samples=self.test_minibatch, mode=self.mode, profile=self.profile,params = self.old_params)
 
        test_fn = test_classifier.build_test_function(self.mode, self.profile)
        validation_cost = 0
        n_words = 0
        unk_words = 0
        fout = open(outfile,'w')
	unk_id = WordID['<UNK>']
	test_batches = test_sample / self.test_minibatch 
	print >> sys.stderr, "test minibatches:",self.test_minibatch 
	
        for ibatch in xrange(test_batches): 
            probs,last_hidden = test_fn(test_x[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch] ,
					test_y[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch],self.hidden_init)
            self.hidden_init = last_hidden

            for y,logp in zip(test_y[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch],probs):
		
            	if y!= unk_id:
                     validation_cost = validation_cost + logp
                     n_words = n_words + 1
		     print >> fout,math.pow(10,logp)
                else:
                     unk_words = unk_words + 1
		     print >> fout, 0
	
	# few samples remain 
	if test_batches * self.test_minibatch < test_sample:
	    residual_samples = test_sample - test_batches * self.test_minibatch
	    print >> sys.stderr, "residual sampes:", residual_samples, test_sample, self.test_minibatch 
	    if useWf:
	    	test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in, n_hidden=self.n_hidden,
                                   n_out=self.n_out, samples=residual_samples, mode=self.mode, profile=self.profile,params = self.old_params,n_features=self.n_features)
	    else:
                test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in+self.n_features, n_hidden=self.n_hidden,
                                   n_out=self.n_out, samples=residual_samples, mode=self.mode, profile=self.profile,params = self.old_params)

            test_fn = test_classifier.build_test_function(self.mode, self.profile)
	    probs,last_hidden = test_fn(test_x[test_batches*self.test_minibatch : test_sample ] ,
                                        test_y[test_batches*self.test_minibatch : test_sample ],self.hidden_init)

            self.hidden_init = last_hidden
	    for y,logp in zip(test_y[test_batches*self.test_minibatch : test_sample],probs):

                if y!= unk_id:
                     validation_cost = validation_cost + logp
                     n_words = n_words + 1
                     print >> fout,math.pow(10,logp)
                else:
                     unk_words = unk_words + 1
                     print >> fout, 0
	

        validation_cost = -1 * validation_cost / numpy.log10(2) / n_words
     	print >> sys.stderr, "\tNumber of unks:",unk_words, "Test Entropy: ", validation_cost
	


if __name__ == '__main__':
    
    #Engine()
    rnn = GpuRNN()
    N = 1000  # number of samples
    n_in = 784 # number of input units
    n_hidden = 100 # number of hidden units
    n_out = 11 # number of output units
    minibatch = 1 # sequence length

    data_x = numpy.random.uniform(size=(N, minibatch, n_in)).astype(theano.config.floatX)
    data_y = numpy.random.uniform(size=(N*minibatch)).astype('int32')
    rnn.train_rnn(data_x,data_y)
    

