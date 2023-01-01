__author__ = 'westbrick'
# class for binary randomized response

import numpy as np
import numpy.random as r
import time
#import discretegauss
import diffprivlib
import utils

class SUCCINCT:
    name = 'SUCCINCT'
    ep = 0.0    # privacy budget epsilon

    d = 0 # domain size + maximum subset size
    m = 0 # maximum subset size
    trate = 0.0 # hit rate when true
    frate = 0.0 # hit rate when false

    # records for consuming time
    clienttime = 0.0
    recordtime = 0.0
    servertime = 0.0

    def __init__(self, d, m, ep, n, base="Laplace"):
        self.ep = ep
        self.d = d
        self.m = m
        self.n = n
        self.base = base
        # records for consuming time
        self.clienttime = 0.0
        self.recordtime = 0.0
        self.servertime = 0.0

        self.__setparams()

    def __setparams(self):
        self.trate = 0.0
        self.frate = 0.0
        # parameter from https://github.com/DPSparseVector/dp-sparse
        self.delta = np.sqrt(self.m*np.log(self.n))/2.5
        # print(np.exp(self.ep/2), self.trate, self.frate)


    def randomizer(self, secrets, seed = None):
        tstart = time.process_time()
        pub = np.zeros(self.d, dtype=int)
        if seed is None:
            seed = r.randint(0, 1000000)*(self.d+self.m)
        v = 0.0
        xs = utils.bitarrayToList(secrets)
        #print("xs", xs)
        for x in xs:
            if x < self.d//2:
                r.seed(seed+x)
                sign = 2*(r.randint(0, 2)-0.5)
                v += sign*1
            elif x < self.d:
                r.seed(seed+x-self.d//2)
                sign = 2*(r.randint(0, 2)-0.5)
                v -= sign*1
        # clip and add noise
        r.seed(None)
        v = max(-self.delta, min(self.delta, v))
        if self.base == "Geometric":
            geometricer = diffprivlib.mechanisms.Geometric(epsilon=self.ep, sensitivity=int(np.ceil(2*self.delta)))
            v = geometricer.randomise(int(v))
            #v += discretegauss.sample_dlaplace(2*self.delta/self.ep)
        elif self.base == "Staircase":
            staircaser = diffprivlib.mechanisms.Staircase(epsilon=self.ep, sensitivity=2*self.delta)
            v = staircaser.randomise(v)
        else:
            # Laplace
            v += r.laplace(0.0, 2*self.delta/self.ep)

        # decode from v
        for i in range(self.d//2):
            r.seed(seed+i)
            sign = 2*(r.randint(0, 2)-0.5)
            pub[i+self.d//2] = sign*v
            #pub[i+self.d//2] = max(-5*self.delta, min(5*self.delta, sign*v))
        self.clienttime += time.process_time()-tstart
        return pub


    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        tstart = time.process_time()
        fs = hits
        self.servertime += time.process_time()-tstart

        d = self.d
        fs[0:d//2] = (fs[0:d//2]+fs[d//2:d])/2
        fs[d//2:d] = fs[0:d//2]-fs[d//2:d]
        return fs[0:d]/n

    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return 0.0