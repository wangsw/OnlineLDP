__author__ = 'westbrick'
# class for binary randomized response

import numpy as np
import scipy as sp
import numpy.random as r
import time

class FutureRand:
    name = 'FUTURERAND'
    ep = 0.0    # privacy budget epsilon

    d = 0 # domain size + maximum subset size
    m = 0 # maximum subset size
    trate = 0.0 # hit rate when true
    frate = 0.0 # hit rate when false

    # records for consuming time
    clienttime = 0.0
    recordtime = 0.0
    servertime = 0.0

    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        # records for consuming time
        self.clienttime = 0.0
        self.recordtime = 0.0
        self.servertime = 0.0

        self.__setparams()

    def __setparams(self):
        self.epnow = self.ep/(5*np.sqrt(self.m))
        self.p = 1/(np.exp(self.epnow)+1)
        self.LB = self.m*self.p-2*np.sqrt(self.m)
        self.UB = self.m/self.epnow*np.log(2*np.exp(self.epnow)/(np.exp(self.epnow)+1))
        self.exceed = 0 # number of outsider


        self.trate = np.exp(self.epnow)/(np.exp(self.epnow)+1)
        self.nrate = 1.0-self.trate
        self.frate = 1.0/2


        self.exceedprob = 1.0-np.sum([sp.special.binom(self.m, r)*np.power(self.trate, self.m-r)*np.power(self.nrate, r) for r in range(max(int(np.ceil(self.LB)), 0), min(int(np.floor(self.UB)), self.m)+1, 1)])
        self.trate = (1-self.exceedprob)*self.trate+self.exceedprob*self.frate
        self.nrate = (1-self.exceedprob)*self.nrate+self.exceedprob*self.frate

        print("futurerand", self.m, self.LB, self.UB, self.exceedprob, self.trate, self.nrate, self.trate+self.nrate-2*self.frate)
        # print(np.exp(self.ep/2), self.trate, self.frate)


    def randomizer(self, secrets):
        tstart = time.process_time()
        pub = np.zeros(self.d, dtype=int)

        d = self.d-self.m
        hamming = 0
        reverses = []
        for i in range(0, d//2):
            p = r.random(1)
            if secrets[i]+secrets[i+d//2] > 0:
                if p < self.trate:
                    pub[i] = pub[i]
                    pub[i+d//2] = pub[i+d//2]
                else:
                    pub[i] = 1-pub[i]
                    pub[i+d//2] = 1-pub[i+d//2]
                    hamming += 1
                    reverses.append(i)
            else:
                if p < self.frate:
                    pub[i] = 1
                    pub[i+d//2] = 0
                else:
                    pub[i] = 0
                    pub[i+d//2] = 1
        if hamming < self.LB or hamming > self.UB:
            self.exceed += 1
            for i in reverses:
                p = r.random(1)
                if p < self.frate:
                    pub[i] = 1
                    pub[i+d//2] = 0
                else:
                    pub[i] = 0
                    pub[i+d//2] = 1
        self.clienttime += time.process_time()-tstart
        return pub

    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        tstart = time.process_time()
        #fs = np.array([(hits[i]-n*self.frate)/(self.trate+self.nrate-2.0*self.frate) for i in range(0, self.d)])
        fs = np.array([0.0 for i in range(0, self.d)])
        us = np.array([hits[i]/(self.trate-self.nrate) for i in range(0, self.d)])
        d = self.d - self.m
        fs[0:d//2] = fs[0:d//2] + fs[d//2:d]
        fs[d//2:d] = us[0:d//2] - us[d//2:d]
        self.servertime += time.process_time()-tstart
        print("number of exceeds", self.exceedprob, self.exceed, n, self.exceed/n)
        self.exceed = 0
        
        fs[0:d//2] = (fs[0:d//2]+fs[d//2:d])/2
        fs[d//2:d] = fs[0:d//2]-fs[d//2:d]
        return fs/n

    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return (self.trate*(1.0-self.trate)+(self.d-1)*self.frate*(1-self.frate))/(n*(self.trate-self.frate)*(self.trate-self.frate))
