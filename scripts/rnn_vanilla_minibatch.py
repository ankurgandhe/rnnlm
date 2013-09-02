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
#---------------------------------------------------------------------------------
class RNN(object):

    #---------------------------------------------------------------------------------
    def __init__(self, rng, output_taps, n_in, n_hidden, n_out, samples, mode, profile, dtype=theano.config.floatX,params=None):
	self.initialize(rng, output_taps, n_in, n_hidden, n_out, samples, mode, profile, dtype,params)

    def initialize(self,rng, output_taps, n_in, n_hidden, n_out, samples, mode, profile, dtype=theano.config.floatX,params=None):
	
        """
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights
            
            :type order: int32
            :param order: order of the RNN (used for higher order RNNs)
            
            :type n_in: int32
            :param n_in: number of input neurons
            
            :type n_hidden: int32
            :param n_hidden: number of hidden units
            
            :type dtype: theano.config.floatX
            :param dtype: theano 32/64bit mode
        """

        # length of output taps
        self.len_output_taps = len(output_taps)
        # input (where first dimension is time)
        self.u = T.matrix() #tensor3()
        # target (where first dimension is time)
        self.t = T.ivector()
        # initial hidden state of the RNN
        self.H = T.vector() # matrix()
        # learning rate
        self.lr = T.scalar()
        self.n_hidden = n_hidden  
        if params == None:
            # recurrent weights as real values
            W = [theano.shared(numpy.random.uniform(size=(n_hidden, n_hidden), low= -.01, high=.01).astype(dtype), 
                               name='W_r' + str(output_taps[u])) for u in range(self.len_output_taps)]
                
            # recurrent bias
            b_h = theano.shared(numpy.zeros((n_hidden,)).astype(dtype), name='b_h')  
            # recurrent activations
            self.h = theano.shared(numpy.zeros((n_hidden,)).astype(dtype), name='h')
                                                        
            # input to hidden layer weights
            W_in = theano.shared(numpy.random.uniform(size=(n_in, n_hidden), low= -.01, high=.01).astype(dtype), name='W_in')
            # input bias
            b_in = theano.shared(numpy.zeros((n_hidden,)).astype(dtype), name='b_in')
        
            # hidden to output layer weights
            W_out = theano.shared(numpy.random.uniform(size=(n_hidden, n_out), low= -.01, high=.01).astype(dtype), name='W_out')
            # output bias
            b_out = theano.shared(numpy.zeros((n_out,)).astype(dtype), name='b_out')      
              
        else:
            # recurrent weights as real values
            W = [theano.shared(params[0],name='W_r' + str(output_taps[u])) for u in range(self.len_output_taps)]

            # recurrent bias
            #b_h = theano.shared(params[1], name='b_h')
            # recurrent activations
            self.h = theano.shared(numpy.zeros((n_hidden,)).astype(dtype), name='h')

            # input to hidden layer weights
            W_in = theano.shared(params[1], name='W_in')
            # input bias
            #b_in = theano.shared(params[3], name='b_in')

            # hidden to output layer weights
            W_out = theano.shared(params[2], name='W_out')
            # output bias
            #b_out = theano.shared(params[5], name='b_out')

        # stack the network parameters            
        self.params = []
        self.params.extend(W)
        #self.params.extend([b_h])
        self.params.extend([W_in]) #([W_in, b_in])
        self.params.extend([W_out]) #([W_out, b_out])
	self.L1  = abs(W[0]).sum() + abs(W_in).sum() + abs(W_out).sum()
	self.L2 = (W[0] ** 2).sum() + (W_in ** 2 ).sum() + ( W_out ** 2).sum() 
	self.lambdaL1 = 0.0 
	self.lambdaL2 = 1e-7
        # the hidden state `h` for the entire sequence, and the output for the
        # entry sequence `y` (first dimension is always time)        
        [h, y], updates = theano.scan(self.step,

                        sequences=self.u,
                        outputs_info=[dict(initial=self.H, taps=[-1]),None],
                        non_sequences=self.params,
			truncate_gradient=5,
                        mode=mode,
                        profile=profile)
         
        # compute the output of the network                        
        # theano has no softmax tensor3() support at the moment
        #y, updates = theano.scan(self.softmax_tensor,
        #            sequences=h,
        #            non_sequences=[W_out, b_out],
        #            mode=mode,
        #            profile=profile)
                      
        # error between output and target
        #self.cost = ((y - self.t) ** 2).sum()                   
        y = y.reshape((samples*1,n_out))        
	#y_tmp = y 
        self.lprob_y_given_x = T.log10(y)[T.arange(self.t.shape[0]), self.t] # T.log((y)[T.arange(self.t.shape[0]), self.t])
	
        self.cost = -T.mean(T.log10(y)[T.arange(self.t.shape[0]), self.t]) #+ self.lambdaL1*self.L1 + self.lambdaL2*self.L2 # -T.mean(T.log(y_tmp)[T.arange(self.t.shape[0]), self.t])
    	self.last_hidden = h[samples-1]

    #def logcost(self,t):
	#return ((self.y - t) ** 2).sum() # -T.mean(T.log(self.y)[T.arange(t.shape[0]), t]) #+ self.lambdaL1*self.L1 + self.lambdaL2*self.L2
    #def lprob_y_given_x(self, t):
	#return T.log(self.y)[T.arange(t.shape[0]), t]
    #---------------------------------------------------------------------------------
    def softmax_tensor(self, h, W, b):
        return T.nnet.softmax(T.dot(h, W) + b)
    def symbolic_softmax(self,x):
    	e = T.exp(x)
        z =  T.sum(e)#.dimshuffle(0, 'x')
	if z ==0:
	    z = 1
	return e / z 
  
    #---------------------------------------------------------------------------------

    def step(self, u_t, *args):     
            """
                step function to calculate BPTT
                
                type u_t: T.matrix()
                param u_t: input sequence of the network
                
                type * args: python parameter list
                param * args: this is needed to implement a more general model of the step function
                             see theano@users: http: // groups.google.com / group / theano - users / \
                             browse_thread / thread / 2fa44792c9cdd0d5
                
            """        
 
            # get the recurrent activations                
            r_act_vals = [args[u] for u in xrange(self.len_output_taps)]
                                        
            # get the recurrent weights
            r_weights = [args[u] for u in range(self.len_output_taps, (self.len_output_taps) * 2)]  
                        
            # get the input/output weights      
            #b_h = args[self.len_output_taps * 2]
            W_in = args[self.len_output_taps * 2] #W_in = args[self.len_output_taps * 2 + 1]
            #b_in = args[self.len_output_taps * 2 + 2]

	    W_out = args[self.len_output_taps * 2 + 1] #W_out = args[self.len_output_taps * 2 + 3]
            #b_out = args[self.len_output_taps * 2 + 4]
            
            # sum up the recurrent activations                                               
            act = T.dot(r_act_vals[0], r_weights[0]) #+ b_h
            for u in xrange(1, self.len_output_taps):   
                act += T.dot(r_act_vals[u], r_weights[u]) #+ b_h
            
            # compute the new recurrent activation
            h_t = T.nnet.sigmoid(T.dot(u_t, W_in) +act ) #+ b_in + act)
            y_t = self.symbolic_softmax(T.dot(h_t, W_out)) # T.nnet.softmax(T.dot(h_t, W_out)) # + b_out)                     

            return h_t,y_t 
            
    #---------------------------------------------------------------------------------
    def build_finetune_functions(self, learning_rate, mode, profile):
        
        print >> sys.stderr, 'Compiling training function'        
        #-----------------------------------------
        # THEANO train function
        #-----------------------------------------           
	#define cost 
	#self.cost = self.logcost(self.t) + + self.lambdaL1*self.L1 + self.lambdaL2*self.L2 

        gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a dictionary
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - self.lr * gparam 
        
        # define the train function    
        train_fn = theano.function([self.u,self.t,self.H], #[sandbox.cuda.basic_ops.gpu_from_host(self.u), sandbox.cuda.basic_ops.gpu_from_host(T.cast(self.t,'float32'))],                             
                             outputs=[self.cost,self.last_hidden],
                             updates=updates,
                             givens={#self.H:T.cast(self.h, 'float32'), 
                                     self.lr:T.cast(learning_rate, 'float32')},
			     #on_unused_input='warn',
                             mode=mode,
                             profile=profile)

        return train_fn

    #---------------------------------------------------------------------------------
    def build_test_function(self,mode,profile):
        print >> sys.stderr, "Compiling test function" 
        #self.h = theano.shared(numpy.zeros((self.n_hidden,)).astype(theano.config.floatX), name='h')
        test_fn = theano.function([self.u, self.t,self.H],
                                  outputs=[self.lprob_y_given_x,self.last_hidden],
                                  #givens={self.H:T.cast(self.h, 'float32')},
                                  mode=mode,
                                  profile=profile)

        return test_fn

    #---------------------------------------------------------------------------------
    def build_get_params(self,mode,profile):
        print >> sys.stderr, "Compiling get params function"
        params_fn = theano.function([],
                                  outputs=self.GetParams(),
                                  mode=mode,
                                  profile=profile)
        return params_fn 
    
    #---------------------------------------------------------------------------------
    def initialize_hidden(self, h_init):
	self.h = theano.shared( h_init, name='h') 
    def GetParams(self):
        return self.params 


#---------------------------------------------------------------------------------

class MetaRNN(object):

    def __init__(self,n_in=784,n_hidden=100,n_out=11,samples=1000,
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
        self.minibatch = minibatch # sequence length
        self.lr = learning_rate 
        self.n_epochs  = n_epochs
	self.output_taps = output_taps 
        self.old_params = old_params
	self.samples = samples 
	self.test_minibatch = self.minibatch*10
        self.hidden_init = numpy.zeros((n_hidden,),dtype=theano.config.floatX) #theano.shared(numpy.zeros((n_hidden,)).astype(theano.config.floatX), name='h')
        print >> sys.stderr, 'network: n_in:{},n_hidden:{},n_out:{},output:softmax'.format(n_in, n_hidden, n_out)
        print >> sys.stderr, 'data: samples:{},batch_size:{}'.format(samples,self.minibatch)
        
        #-----------------------------------------
        # RNN SETUP
        #-----------------------------------------
        # initialize random generator
        self.rng = numpy.random.RandomState(1234)
        # construct the CTC_RNN class
        self.classifier = RNN(rng=self.rng, output_taps=output_taps, n_in=n_in, n_hidden=n_hidden, n_out=n_out, samples=self.N, mode=self.mode, profile=self.profile,params=old_params)
        # fetch the training function
        #self.train_fn = self.classifier.build_finetune_functions(self.lr, self.mode, self.profile)

    def train_rnn(self,data_x,data_y,valid_x=None,valid_y=None,gpu_copy_size=10000,unk_id=2,fout=""):
        #--------------------------------------------
        # Shape of X x= Nx1xn_in , shape of y : N
        #--------------------------------------------

        self.train_fn = self.classifier.build_finetune_functions(self.lr, self.mode, self.profile)
	if valid_x == None or valid_y == None:
	    print >> sys.stderr, "No validation set provided. Running till set number of epochs" 
	    test_fn = None 
	else:
	    test_sample = valid_x.shape[0]
	    test_batches = test_sample / self.test_minibatch 
	    test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in, n_hidden=self.n_hidden, n_out=self.n_out, samples=self.test_minibatch, mode=self.mode, profile=self.profile,params = self.old_params)
	    test_fn = test_classifier.build_test_function(self.mode, self.profile)

	get_params = self.classifier.build_get_params(self.mode,self.profile);	
        best_validation_cost = 1e10 ;

        print >> sys.stderr, 'Running ({} epochs), each with {} batches'.format(self.n_epochs,self.n_batches)
        start_time = time.clock()
	gpu_copy_size = self.samples + 1 
  	n_gpu_batches = self.samples / gpu_copy_size +1 ; 
	#print >> sys.stderr, n_gpu_batches 
        for iepoch in xrange(self.n_epochs) :
	    print >> sys.stderr, "Epoch:",iepoch
	    total_cost = 0
            self.hidden_init = numpy.zeros((self.n_hidden,),dtype=theano.config.floatX)
	    self.lr = self.lr / 1.5 
            self.train_fn = self.classifier.build_finetune_functions(self.lr, self.mode, self.profile)
	    for gbatch in xrange(n_gpu_batches):
	    	data_x_shared = data_x[gbatch*gpu_copy_size:min(self.samples,(gbatch+1)*gpu_copy_size)]
	    	data_y_shared = data_y[gbatch*gpu_copy_size:min(self.samples,(gbatch+1)*gpu_copy_size)]
		n_batches = len(data_y[gbatch*gpu_copy_size:min(self.samples,(gbatch+1)*gpu_copy_size)])/self.N 
            	for ibatch in range(n_batches):
                    data_x1 = data_x_shared[ibatch*self.N : (ibatch+1)*self.N]
                    data_y1 = data_y_shared[ibatch*self.N : (ibatch+1)*self.N] 
                    batch_cost,final_hidden  = self.train_fn(data_x1, data_y1,self.hidden_init)
		    self.hidden_init = final_hidden 
		    #self.classifier.initialize_hidden(final_hidden) 
                    total_cost = total_cost + batch_cost 
                    if (ibatch%5000)==0:
                    	print >> sys.stderr , "Current Train entropy:", total_cost/numpy.log10(2)/((ibatch+1)) #Total Batch",ibatch+1,"cost", batch_cost
            
	    self.learnt_params = get_params()
	    self.hidden_init = numpy.zeros((self.n_hidden,),dtype=theano.config.floatX)
            test_classifier.initialize(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in, n_hidden=self.n_hidden, n_out=self.n_out, samples=self.test_minibatch, mode=self.mode, profile=self.profile,params = self.learnt_params)
	    if test_fn != None:
	        test_fn = test_classifier.build_test_function(self.mode, self.profile)
		validation_cost = 0 
		n_words = 0 
		unk_words = 0 
		for ibatch in xrange(test_batches):
	    	    probs,last_hidden = test_fn(valid_x[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch] ,valid_y[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch],self.hidden_init)
		    self.hidden_init = last_hidden
		   
		    for y,logp in zip(valid_y[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch],probs):
			if y!= unk_id:
			    validation_cost = validation_cost + logp 
			    n_words = n_words + 1 
		    	else:
			    unk_words = unk_words + 1 
		    #validation_cost = validation_cost +numpy.mean(-1*probs) 
	    else:
	    	probs = 0 
		validation_cost = 0 
	    validation_cost = -1 * validation_cost / numpy.log10(2) / n_words 
	    #validation_cost = validation_cost / numpy.log10(2)/test_batches 
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
	test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in, n_hidden=self.n_hidden, n_out=self.n_out, samples=test_sample, mode=self.mode, profile=self.profile,params = self.old_params)
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
        test_classifier = RNN(rng=self.rng, output_taps=self.output_taps, n_in=self.n_in, n_hidden=self.n_hidden, 
				   n_out=self.n_out, samples=self.test_minibatch, mode=self.mode, profile=self.profile,params = self.old_params)
        test_fn = test_classifier.build_test_function(self.mode, self.profile)
        validation_cost = 0
        n_words = 0
        unk_words = 0
        fout = open(outfile,'w')
	unk_id = WordID['<UNK>']
	test_batches = test_sample / self.test_minibatch 
        for ibatch in xrange(test_batches): 
            probs,last_hidden = test_fn(test_x[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch] ,test_y[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch],self.hidden_init)
            self.hidden_init = last_hidden

            for y,logp in zip(test_y[ibatch*self.test_minibatch : (ibatch+1)*self.test_minibatch],probs):
		
            	if y!= unk_id:
                     validation_cost = validation_cost + logp
                     n_words = n_words + 1
		     print >> fout,math.pow(10,logp)
                else:
                     unk_words = unk_words + 1
		     print >> fout, 0
                    #validation_cost = validation_cost +numpy.mean(-1*probs)
        validation_cost = -1 * validation_cost / numpy.log10(2) / n_words
            #validation_cost = validation_cost / numpy.log10(2)/test_batches
     	print >> sys.stderr, "\tNumber of unks:",unk_words, "Test Entropy: ", validation_cost



if __name__ == '__main__':
    
    #Engine()
    rnn = MetaRNN()
    N = 1000  # number of samples
    n_in = 784 # number of input units
    n_hidden = 100 # number of hidden units
    n_out = 11 # number of output units
    minibatch = 1 # sequence length

    data_x = numpy.random.uniform(size=(N, minibatch, n_in)).astype(theano.config.floatX)
    data_y = numpy.random.uniform(size=(N*minibatch)).astype('int32')
    rnn.train_rnn(data_x,data_y)
    

