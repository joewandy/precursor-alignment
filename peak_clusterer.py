
import numpy as np
from scipy.special import gammaln
from scipy.stats import beta
import pylab as plt

class shape_cluster_gibbs(object):

    def __init__(self, corr_mat, hyper_pars, n_samples=100, n_burn=10, K=30, seed=12345):
        self.corr_mat = corr_mat
        self.hyper_pars = hyper_pars
        self.n_samples = n_samples
        self.n_burn = n_burn
        self.K = K
        self.seed = seed
        np.random.seed(self.seed)
        self.n_peaks = (corr_mat.shape)[0]
        self._create_like_mats()

    def _initialise_clustering(self):
        # put all into the first cluster
        self.Z = np.zeros(self.n_peaks, dtype=np.int64)
        self.counts = np.zeros(self.K).tolist()
        self.counts[0] = self.n_peaks

    def _sample(self):
        base_like = self.out_like.sum(axis=0)
        for samp in np.arange(self.n_samples): # for each sample

            if samp%10==0:
                print "Sample " + str(samp)

            for peak in np.arange(self.n_peaks):

                # remove from model
                this_peak = peak
                this_cluster = self.Z[this_peak]
                self.Z[this_peak] = -1
                self.counts[this_cluster] -= 1

                # compute prior and likelihood
                prior = np.log([c + self.hyper_pars.conc_par/self.K for c in self.counts])
                like = np.zeros_like(prior)
                for k in np.arange(self.K):
                    in_pos = np.where(self.Z==k)[0]
                    like[k] = base_like[this_peak] - self.out_like[this_peak,in_pos].sum()
                    like[k] += self.in_like[this_peak,in_pos].sum()

                # compute posterior and resample a new cluster assignment
                post = prior + like
                post = np.exp(post - np.max(post))
                post /= post.sum()
                new_cluster = np.where(np.random.rand()<post.cumsum())[0][0]

                # put back into the model
                self.Z[this_peak] = new_cluster
                self.counts[new_cluster] += 1

    def _create_like_mats(self):
        print "Creating likelihood matrices"
        self.in_like = np.zeros((self.n_peaks,self.n_peaks))
        self.out_like = np.zeros((self.n_peaks,self.n_peaks))
        for n in np.arange(self.n_peaks-1):
            for m in np.arange(n+1,self.n_peaks): # pre-computes the in and out likelihood (eq. 2 in Tommi's paper)
                if self.corr_mat[n,m]!=0:
                    in_val = np.log(self.hyper_pars.in_prob) + log_beta_pdf(self.corr_mat[n,m],self.hyper_pars.in_alpha,self.hyper_pars.in_beta)
                    out_val = np.log(self.hyper_pars.out_prob) + log_beta_pdf(self.corr_mat[n,m],self.hyper_pars.out_alpha,self.hyper_pars.out_beta)
                else:
                    in_val = np.log(1-self.hyper_pars.in_prob)
                    out_val = np.log(1-self.hyper_pars.out_prob)

                self.in_like[n][m] = in_val
                self.in_like[m][n] = in_val
                self.out_like[n][m] = out_val
                self.out_like[m][n] = out_val

    def __repr__(self):
        return "Peak shape clusterer"

def log_beta_pdf(x,a,b):
    o = gammaln(a + b) - gammaln(a) - gammaln(b)
    o = o + (a-1)*np.log(x) + (b-1)*np.log(1-x)
    return o

class hyper(object):
    in_alpha = 10.0
    out_alpha = 1.0
    in_beta = 1.0
    out_beta = 10.0
    conc_par = 5.0
    in_prob = 0.99
    out_prob = 0.1

class data_generator(object):
    def __init__(self,hyper,n_peaks = 100,seed = 123):
        self.hyper_pars = hyper
        self.n_peaks = n_peaks
        self.seed = seed
        np.random.seed(self.seed)

    def _make_data(self):
        self.Z = []
        self.counts = []
        self.Z.append(1)
        self.counts.append(1)
        for n in np.arange(1,self.n_peaks):
        #     List concatanation
            temp = np.array(self.counts + [self.hyper_pars.conc_par])
            temp = temp/temp.sum()
            prob = temp.cumsum()
            pos = np.nonzero(np.random.rand()<prob)[0][0]
            self.Z.append(pos)
            if pos >= len(self.counts):
                self.counts.append(1)
            else:
                self.counts[pos] += 1

        self.Z = np.sort(self.Z)
        self.K = np.max(self.Z)
        self.intensities = []
        for n in np.arange(self.n_peaks):
            self.intensities.append(np.random.rand())
        self.corr_mat = np.zeros((self.n_peaks,self.n_peaks))
        for n in np.arange(self.n_peaks-1):
            this_cluster = self.Z[n]
            for m in np.arange(n+1,self.n_peaks):
                if self.Z[m] == this_cluster:
                    if np.random.rand() < self.hyper_pars.in_prob:
                        this_val = beta.rvs(self.hyper_pars.in_alpha,self.hyper_pars.in_beta)
                    else:
                        this_val = 0
                else:
                    if np.random.rand() < self.hyper_pars.out_prob:
                        this_val = beta.rvs(self.hyper_pars.out_alpha,self.hyper_pars.out_beta)
                    else:
                        this_val = 0

                self.corr_mat[n,m] = this_val
                self.corr_mat[m,n] = this_val