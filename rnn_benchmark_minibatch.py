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

#---------------------------------------------------------------------------------
class RNN(object):

    #---------------------------------------------------------------------------------
    def __init__(self, rng, output_taps, n_in, n_hidden, n_out, samples, minibatch, mode, profile, dtype=theano.config.floatX,params=None):
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
        self.u = T.tensor3()
        # target (where first dimension is time)
        self.t = T.ivector()
        # initial hidden state of the RNN
        self.H = T.matrix()
        # learning rate
        self.lr = T.scalar()
        
        if params == None:
            # recurrent weights as real values
            W = [theano.shared(numpy.random.uniform(size=(n_hidden, n_hidden), low= -.01, high=.01).astype(dtype), 
                               name='W_r' + str(output_taps[u])) for u in range(self.len_output_taps)]
                
            # recurrent bias
            b_h = theano.shared(numpy.zeros((n_hidden,)).astype(dtype), name='b_h')  
            # recurrent activations
            self.h = theano.shared(numpy.zeros((minibatch, n_hidden)).astype(dtype), name='h')
                                                        
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
            b_h = theano.shared(params[1], name='b_h')
            # recurrent activations
            self.h = theano.shared(numpy.zeros((minibatch, n_hidden)).astype(dtype), name='h')

            # input to hidden layer weights
            W_in = theano.shared(params[2], name='W_in')
            # input bias
            b_in = theano.shared(params[3], name='b_in')

            # hidden to output layer weights
            W_out = theano.shared(params[4], name='W_out')
            # output bias
            b_out = theano.shared(params[5], name='b_out')

        # stack the network parameters            
        self.params = []
        self.params.extend(W)
        self.params.extend([b_h])
        self.params.extend([W_in, b_in])
        self.params.extend([W_out, b_out])
  
        # the hidden state `h` for the entire sequence, and the output for the
        # entry sequence `y` (first dimension is always time)        
        h, updates = theano.scan(self.step,
                        sequences=self.u,
                        outputs_info=dict(initial=self.H, taps=[-1]),
                        non_sequences=self.params,
                        mode=mode,
                        profile=profile)
        
        # compute the output of the network                        
        # theano has no softmax tensor3() support at the moment
        y, updates = theano.scan(self.softmax_tensor,
                    sequences=h,
                    non_sequences=[W_out, b_out],
                    mode=mode,
                    profile=profile)
                      
        # error between output and target
        #self.cost = ((y - self.t) ** 2).sum()                   
        y_tmp = y.reshape((samples*minibatch,n_out))        

        self.lprob_y_given_x = T.log(y_tmp)[T.arange(self.t.shape[0]), self.t]

        self.cost = -T.mean(T.log(y_tmp)[T.arange(self.t.shape[0]), self.t])
    
    #---------------------------------------------------------------------------------
    def softmax_tensor(self, h, W, b):
        return T.nnet.softmax(T.dot(h, W) + b)
      
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
            b_h = args[self.len_output_taps * 2]
            W_in = args[self.len_output_taps * 2 + 1]
            b_in = args[self.len_output_taps * 2 + 2]
            
            # sum up the recurrent activations                                               
            act = theano.dot(r_act_vals[0], r_weights[0]) + b_h
            for u in xrange(1, self.len_output_taps):   
                act += T.dot(r_act_vals[u], r_weights[u]) + b_h
            
            # compute the new recurrent activation
            h_t = T.tanh(T.dot(u_t, W_in) + b_in + act)
                             
            return h_t
            
    #---------------------------------------------------------------------------------
    def build_finetune_functions(self, learning_rate, mode, profile):
        
        print 'Compiling'        
        #-----------------------------------------
        # THEANO train function
        #-----------------------------------------           
        gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a dictionary
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - self.lr * gparam 
        
        # define the train function    
        train_fn = theano.function([self.u, self.t],                             
                             outputs=self.cost,
                             updates=updates,
                             givens={self.H:T.cast(self.h, 'float32'), 
                                     self.lr:T.cast(learning_rate, 'float32')},
                             mode=mode,
                             profile=profile)

        return train_fn

    #---------------------------------------------------------------------------------
    def build_test_function(self,mode,profile):
        print >> sys.stderr, "Compiling test function" 
        test_fn = theano.function([self.u, self.t],
                                  outputs=self.lprob_y_given_x,
                                  givens={self.H:T.cast(self.h, 'float32')},
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
    def GetParams(self):
        return self.params 


#---------------------------------------------------------------------------------
class Engine(object):

    def __init__(self,
                learning_rate=0.01,
                n_epochs=13,
                output_taps=[-1]):

        #-----------------------------------------
        # BENCHMARK SETUP
        #----------------------------------------- 
        """
            Please note that if you are comparing different RNNs (C/C++/TORCH/...)
            the data & network topology and parameters should be at least the same, to
            be fair.
            
            f.e. samples, batchsize, learning_rate, epochs, #neurons, ... (other obvious stuff)
        """
        
        #-----------------------------------------
        # THEANO SETUP
        #-----------------------------------------
        # setup mode
        mode = theano.Mode(linker='cvm')
        # setup profile          
        profile = 0      
        
        #-----------------------------------------
        # MODEL SETUP
        #-----------------------------------------
        N = 1000  # number of samples
        n_in = 784 # number of input units
        n_hidden = 100 # number of hidden units        
        n_out = 11 # number of output units        
        minibatch = 1 # sequence length 
        print 'network: n_in:{},n_hidden:{},n_out:{},output:softmax'.format(n_in, n_hidden, n_out)
        print 'data: samples:{},length:{},batch_size:{}'.format(N,minibatch,N)
        
        # create data vectors    
        # TODO: fix reshape, ValueError: ('setting an array element with a sequence.', 'Bad input argument to theano)
        #data_x = theano.shared(numpy.random.uniform(size=(N, minibatch, n_in)).astype(theano.config.floatX))
        #data_y = theano.shared(numpy.random.uniform(size=(N*minibatch)).astype(theano.config.floatX))     
        data_x = numpy.random.uniform(size=(N, minibatch, n_in)).astype(theano.config.floatX)
        data_y = numpy.random.uniform(size=(N*minibatch)).astype('int32')


        #-----------------------------------------
        # RNN SETUP
        #-----------------------------------------           
        # initialize random generator                                                  
        rng = numpy.random.RandomState(1234)      
        # construct the CTC_RNN class
        classifier = RNN(rng=rng, output_taps=output_taps, n_in=n_in, n_hidden=n_hidden, n_out=n_out, samples=N, minibatch=minibatch, mode=mode, profile=profile)    
        # fetch the training function
        train_fn = classifier.build_finetune_functions(learning_rate, mode, profile)   
        
        #-----------------------------------------
        # BENCHMARK START
        #-----------------------------------------                                                    
        # start the benchmark    
        print 'Running ({} epochs)'.format(n_epochs)        
        start_time = time.clock()     
        for _ in xrange(n_epochs) :             
            batch_cost  = train_fn(data_x, data_y)                             
	    print >> sys.stderr , "Total Batch cost", batch_cost 

        print >> sys.stderr, ('     overall training epoch time (%.5fm)' % ((time.clock() - start_time) / 60.))
  
        #-----------------------------------------
        # TEST RNN SETUP
        #-----------------------------------------

        # Get trained parameters of classifier 
        get_params = classifier.build_get_params(mode,profile);
        learnt_params = get_params()

        #Create Test data. Random for now.
        N_test = 100
        data_test_x = numpy.random.uniform(size=(N_test, minibatch, n_in)).astype(theano.config.floatX)
        data_test_y = numpy.random.uniform(size=(N_test*minibatch)).astype('int32')

        Testclassifier = RNN(rng=rng, output_taps=output_taps, n_in=n_in, n_hidden=n_hidden, n_out=n_out, samples=N_test, minibatch=minibatch, mode=mode, profile=profile,params=learnt_params)
        print >> sys.stderr, "Testing RNN "
        test_fn = Testclassifier.build_test_function(mode, profile)
        probs = test_fn(data_test_x,data_test_y)
        print >> sys.stderr, probs
        
#---------------------------------------------------------------------------------    


class MetaRNN(object):

    def __init__(self,n_in=784,n_hidden=100,n_out=11,samples=1000,
                learning_rate=0.1,minibatch=1,
                n_epochs=10,
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

        self.N = samples  # number of samples
        self.n_in = n_in # number of input units
        self.n_hidden = n_hidden # number of hidden units
        self.n_out = n_out # number of output units
        self.minibatch = minibatch # sequence length
        self.lr = learning_rate 
        self.n_epochs  = n_epochs
	self.output_taps = output_taps 
        print 'network: n_in:{},n_hidden:{},n_out:{},output:softmax'.format(n_in, n_hidden, n_out)
        print 'data: samples:{},length:{},batch_size:{}'.format(samples,self.minibatch,samples)
        
        #-----------------------------------------
        # RNN SETUP
        #-----------------------------------------
        # initialize random generator
        rng = numpy.random.RandomState(1234)
        # construct the CTC_RNN class
        self.classifier = RNN(rng=rng, output_taps=output_taps, n_in=n_in, n_hidden=n_hidden, n_out=n_out, samples=self.N, minibatch=self.minibatch, mode=self.mode, profile=self.profile)
        # fetch the training function
        self.train_fn = self.classifier.build_finetune_functions(self.lr, self.mode, self.profile)

    def train_rnn(self,data_x,data_y,valid_x=None,valid_y=None):
        #--------------------------------------------
        # Shape of X x= Nx1xn_in , shape of y : N
        #--------------------------------------------
        
	if valid_x == None or valid_y == None:
	    print >> sys.stderr, "No validation set provided. Running till set number of epochs" 
	    test_fn = None 
	else:
	    test_sample = valid_x.shape[0]
	    test_classifier = RNN(rng=rng, output_taps=self.output_taps, n_in=self.n_in, n_hidden=self.n_hidden, n_out=self.n_out, samples=test_sample, minibatch=self.minibatch, mode=self.mode, profile=self.profile)
	    test_fn = Testclassifier.build_test_function(mode, profile)

        print 'Running ({} epochs)'.format(self.n_epochs)
        start_time = time.clock()
        for _ in xrange(self.n_epochs) :
            batch_cost  = self.train_fn(data_x, data_y)
            probs = test_fn(data_test_x,data_test_y)
            print >> sys.stderr , "Total Batch cost", batch_cost


        get_params = self.classifier.build_get_params(self.mode,self.profile);
        self.learnt_params = get_params()




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
    
