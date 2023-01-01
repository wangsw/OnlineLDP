__author__ = 'westbrick'

# class for ExSub mechanism

import numpy as np
import numpy.random as r
from scipy.special import comb
import time

import utils
import decimal
from decimal import Decimal as D

decimal.getcontext().prec = 32


class EXSUB:
    name = 'EXSUB'
    ep = 0.0  # privacy budget epsilon

    d = 0  # domain size + maximum subset size
    m = 0  # maximum subset size
    trate = 0.0  # hit rate when true
    frate = 0.0  # hit rate when false
    normalizer = 0.0  # normalizer for proportional probabilities

    # records for consuming time
    clienttime = 0.0
    recordtime = 0.0
    servertime = 0.0

    def __init__(self, d, m, ep, k=None, alpha=0.1, clip=False):
        self.ep = ep
        self.d = d
        self.m = m
        self.k = k
        self.alpha = alpha
        self.clip = clip

        self.clienttime = 0.0
        self.recordtime = 0.0
        self.servertime = 0.0

        self.__setparams()

    def __setparams(self):
        if self.k == None:
            self.k = self.bestSubsetSize(self.d, self.m, self.ep)[0]
        d = self.d//2  # length of {+1,-1,0} vector
        m = self.m
        k = self.k
        nonInterCount = 0
        for j in range(k+1):
            nonInterCount += D(utils.Comb(m, j)*utils.Comb(d-m, k-j)*(D(2.0)**(k-j)))
        interCount = D(utils.Comb(d, k)*(D(2.0)**k)) - nonInterCount
        normalizer = nonInterCount + interCount * D(np.exp(self.ep))
        self.normalizer = normalizer

        self.trate = float(D(utils.Comb(d-1, k-1)*(D(2.0)**(k-1)))*D(np.exp(self.ep))/normalizer)

        nonInterCount = 0
        for j in range(k):
            nonInterCount += D(utils.Comb(m-1, j)*utils.Comb(d-m, k-1-j)*(D(2.0)**(k-1-j)))
        self.nrate = float((nonInterCount+(D(utils.Comb(d-1, k-1)*(D(2.0)**(k-1)))-nonInterCount)*D(np.exp(self.ep)))/normalizer)

        nonInterCount = 0
        for j in range(k):
            nonInterCount += D(utils.Comb(m, j)*utils.Comb(d-m-1, k-1-j)*(D(2.0)**(k-1-j)))
        self.frate = float((nonInterCount+(D(utils.Comb(d-1, k-1)*(D(2.0)**(k-1)))-nonInterCount)*D(np.exp(self.ep)))/normalizer)

        # pre-computed data structure for sampling
        probs = np.full((k+1,k+1), 0.0, dtype=float)
        for inter in range(0, k + 1):
            for ninter in range(k-inter+1):
                probs[inter,ninter] = D(utils.Comb(m, inter)*utils.Comb(m-inter, ninter))*D(utils.Comb(d-m, k-inter-ninter)*(D(2.0)**(k-inter-ninter)))/self.normalizer
                #sums += D(utils.Comb(m, inter)*utils.Comb(m-inter, ninter))*D(utils.Comb(d-m, k-inter-ninter)*D(2.0**(k-inter-ninter)))/self.normalizer
                probs[inter,ninter] = float(probs[inter,ninter])
        probs = probs*np.exp(self.ep)
        probs[0] = probs[0]/np.exp(self.ep)
        self.probs = probs
        # print(np.exp(self.ep/2), self.trate, self.frate)
        print("ExSub (k, trate, nrate, frate)", k, self.trate, self.nrate, self.frate, np.sum(probs))

    @staticmethod
    def bestSubsetSize(d, m, ep, alpha=0.1):
        od = d
        d = d//2
        errorbounds = [0.0]*od
        infos = [None]*od
        ep = D(ep)
        #print("kvsm params", d, m, 1, int((d+m)/max(m-1, 1+np.sqrt(m))))
        end = min(d, int(d//2)+m)
        if (d+m) > 300:
            # handling huge combinatorics
            end = min(d, int((d+m)/max(m-1, 1+np.sqrt(m)))+m)
        for k in range(1, end):
            nonInterCount = 0
            for j in range(k+1):
                nonInterCount += D(utils.Comb(m, j)*utils.Comb(d-m, k-j)*(D(2.0)**(k-j)))
            interCount = D(utils.Comb(d, k)*(D(2.0)**k)) - nonInterCount
            normalizer = nonInterCount + interCount * D(np.exp(ep))

            trate = float(D(utils.Comb(d-1, k-1)*(D(2.0)**(k-1)))*D(np.exp(ep))/normalizer)

            nonInterCount = 0
            for j in range(k):
                nonInterCount += D(utils.Comb(m-1, j)*utils.Comb(d-m, k-1-j)*(D(2.0)**(k-1-j)))
            nrate = float((nonInterCount+(D(utils.Comb(d-1, k-1)*(D(2.0)**(k-1)))-nonInterCount)*D(np.exp(ep)))/normalizer)

            nonInterCount = 0
            for j in range(k):
                nonInterCount += D(utils.Comb(m, j)*utils.Comb(d-m-1, k-1-j)*(D(2.0)**(k-1-j)))
            frate = float((nonInterCount+(D(utils.Comb(d-1, k-1)*(D(2.0)**(k-1)))-nonInterCount)*D(np.exp(ep)))/normalizer)

            #print("kvsm (d, m, k, trate, nrate, frate)", d, m, k, trate, nrate, frate)

            #minusVar = (m*trate*(1-trate)+m*nrate*(1-nrate)+(od-2*m)*frate*(1-frate))/((trate-nrate)*(trate-nrate))
            minusVar = (m*(trate+nrate-(trate-nrate)**2)+(d-m)*2*frate)/((trate-nrate)*(trate-nrate))
            plusVar = m*(trate+nrate)*(1-trate-nrate)+(d-m)*2*frate*(1-2*frate)/((trate+nrate-2*frate)*(trate+nrate-2*frate))
            errorbounds[k] = minusVar+plusVar*alpha
            infos[k] = [trate, nrate, frate, errorbounds[k]]
            #print("kvsm (k, trate, nrate, frate, weighted error)", k, trate, nrate, frate, errorbounds[k])
        bestk = np.argmin(errorbounds[1: end]) + 1
        #bestk = max(1, bestk-2)
        #print("kvsm best ", bestk, infos[bestk], errorbounds)
        return [bestk] + infos[bestk]

    def randomizer(self, secrets):
        tstart = time.process_time()
        #print(secrets, len(secrets), self.d)
        assert len(secrets) == self.d
        pub = np.zeros(self.d, dtype=int)
        # probs = self.probs
        # probs = np.copy(self.probs)

        # print(self.accprobs)
        p = r.random(1)[0]
        # print("p", p, probs, secrets.tolist())
        sinter = 0
        sninter = 0
        for inter in range(0, self.k + 1):
            for ninter in range(self.k-inter+1):
                p -= self.probs[inter, ninter]
                if p < 0.0:
                    sinter = inter
                    sninter = ninter
                    break
            if p < 0.0:
                break
        assert np.max(secrets[0:self.d//2]+secrets[self.d//2:self.d]) <= 1
        xs = list(np.nonzero(secrets)[0]) #utils.bitarrayToList(secrets)
        r.shuffle(xs)
        pubset = xs[:sinter] # utils.reservoirsample(xs, sinter)
        nxs = [x+self.d//2 if x < self.d//2 else x-self.d//2 for x in xs if x not in pubset]
        r.shuffle(nxs)
        pubset += nxs[:sninter] #utils.reservoirsample(nxs, sninter)
        zeros = [x for x in range(0, self.d//2) if x not in xs and x+self.d//2 not in xs]
        r.shuffle(zeros)
        for zero in zeros[:self.k-sinter-sninter]: #utils.reservoirsample(zeros, self.k-sinter-sninter):
            if r.random() < 0.5:
                pubset.append(zero)
            else:
                pubset.append(zero+self.d//2)
        # print("pubset", pubset)
        pub[pubset] = 1
        if np.max(pub[0:self.d//2]+pub[self.d//2:self.d]) > 1:
            print("ExSub debug", xs, nxs, zeros, pub[0:self.d//2])
            print("ExSub debug", xs, nxs, zeros, pub[self.d//2:self.d])
        self.clienttime += time.process_time() - tstart

        return pub

    def decoder(self, hits, n):
        # debias hits but without projecting to simplex
        # print('rates', self.trate, self.frate)
        tstart = time.process_time()
        fs = np.array([(hits[i]-n*self.frate)/(self.trate+self.nrate-2.0*self.frate) for i in range(0, self.d)])
        us = np.array([hits[i]/(self.trate-self.nrate) for i in range(0, self.d)])
        if self.clip:
            delta = np.sqrt(self.m*np.log(n))/(0.7*self.ep)+np.sqrt(self.m*np.log(n))/(2.5)
            clip = min(1/(self.trate-self.frate), delta)
            us = np.array([hits[i]*clip for i in range(0, self.d)])
        d = self.d
        fs[0:d//2] = fs[0:d//2] + fs[d//2:d]
        fs[d//2:d] = us[0:d//2] - us[d//2:d]
        self.servertime += time.process_time() - tstart

        fs[0:d//2] = (fs[0:d//2]+fs[d//2:d])/2
        fs[d//2:d] = fs[0:d//2]-fs[d//2:d]
        return fs[0:d]/n

    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        m = self.m
        d = self.d//2
        od = self.d
        trate = self.trate
        nrate = self.nrate
        frate = self.frate
        minusVar = (m*(trate+nrate-(trate-nrate)**2)+(d-m)*2*frate)/((trate-nrate)*(trate-nrate))
        plusVar = m*(trate+nrate)*(1-trate-nrate)+(d-m)*2*frate*(1-2*frate)/((trate+nrate-2*frate)*(trate+nrate-2*frate))
        return minusVar
