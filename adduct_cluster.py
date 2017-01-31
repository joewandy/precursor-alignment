import sys

from models import Feature
import transformation
import numpy as np
import pylab as plt
from scipy.special import gammaln
from scipy.stats import beta
import collections

class Peak(object):

    def __init__(self,mass,rt,intensity):
        self.mass = mass
        self.rt = rt
        self.intensity = intensity

class Cluster(object):

    def __init__(self, mHPeak, M, id, mass_tol = 5, rt_tol = 10):
        self.mHPeak = mHPeak
        self.N = 1
        self.members = []
        self.mass_sum = M
        self.rt_sum = self.mHPeak.rt
        self.peak_trans = ["M+H"]
        self.M = M
        self.prior_rt_mean = mHPeak.rt
        self.prior_mass_mean = M

        delta = mass_tol*M/1e6
        var = (delta/3.0)**2
        self.prior_mass_precision = 1.0/var
        self.mass_precision = 1.0/var

        delta = rt_tol
        var = (delta/3.0)**2
        self.prior_rt_precision = 1.0/var
        self.rt_precision = 1.0/var

        self.id = id

    def compute_rt_like(self,rt):
        post_prec = self.prior_rt_precision + self.N*self.rt_precision
        post_mean = (1.0/post_prec)*(self.prior_rt_precision*self.prior_rt_mean + self.rt_precision*self.rt_sum)
        pred_prec = (1.0/(1.0/post_prec + 1.0/self.rt_precision))
        self.mu_rt = post_mean
        return -0.5*np.log(2*np.pi) + 0.5*np.log(pred_prec) - 0.5*pred_prec*(rt-post_mean)**2

    def compute_mass_like(self,mass):
        post_prec = self.prior_mass_precision + self.N*self.mass_precision
        post_mean = (1.0/post_prec)*(self.prior_mass_precision*self.prior_mass_mean + self.mass_precision*self.mass_sum)
        pred_prec = (1.0/(1.0/post_prec + 1.0/self.mass_precision))
        self.mu_mass = post_mean
        return -0.5*np.log(2*np.pi) + 0.5*np.log(pred_prec) - 0.5*pred_prec*(mass-post_mean)**2

class Possible(object):

    def __init__(self,cluster,transformation,transformed_mass,rt):
        self.count = 0
        self.cluster = cluster
        self.transformation = transformation
        self.transformed_mass = transformed_mass
        self.rt = rt

class AdductCluster(object):

    def __init__(self, rt_tol = 10, mass_tol = 5, transformation_file = 'pos_transformations_full.yml', alpha = 1,
                corr_mat = None, in_alpha = 10.0, out_alpha = 1.0, in_beta = 1.0, out_beta = 10.0, in_prob = 0.99, out_prob = 0.1,
                verbose = 0, mh_biggest = True):
        self.mass_tol = mass_tol
        self.rt_tol = rt_tol
        self.transformation_file = transformation_file
        self.load_transformations()
        self.alpha = alpha

        self.corr_mat = corr_mat
        self.in_alpha = in_alpha
        self.out_alpha = out_alpha
        self.in_beta = in_beta
        self.out_beta = out_beta
        self.in_prob = in_prob
        self.out_prob = out_prob

        self.verbose = verbose
        self.nSamples = 0
        self.mh_biggest = mh_biggest

    def load_transformations(self):
        self.transformations = transformation.load_from_file(self.transformation_file)
        self.MH = None
        for t in self.transformations:
            print '- %s' % t.name
            if t.name=="M+H":
                self.MH = t

    def init_from_file(self,filename):
        peak_list = []
        with open(filename,'r') as f:
            heads = f.readline()
            for line in f:
                line = line.split('\t')
                mass = float(line[0]);
                rt = float(line[1]);
                intensity = float(line[2]);
                peak_list.append(Peak(mass,rt,intensity))

        if self.verbose:
            print "Loaded {} peaks from {}".format(len(peak_list),filename);
        self.init_from_list(peak_list)

    def init_from_list(self,peak_list):

        self.peaks = []
        self.clusters = []
        self.possible = {}
        self.Z = {}
        self.todo = []
        self.peak_idx = {}
        self.clus_poss = {}
        current_id = 0
        for p in peak_list:
            self.peaks.append(p)
            c = Cluster(p,self.MH.transform(p), current_id,
                mass_tol = self.mass_tol,rt_tol = self.rt_tol)
            c.members.append(p)
            current_id += 1
            self.clusters.append(c)
            poss = Possible(c, self.MH, self.MH.transform(p), p.rt)
            self.possible[p] = {}
            self.possible[p] = [poss]
            self.Z[p] = poss
            self.clus_poss[c] = [poss]

        if self.verbose:
            print "Created {} clusters".format(len(self.clusters))
        self.K = len(self.clusters)

        if self.verbose:
            if self.mh_biggest:
                print "Binning with mh_biggest = True"
            else:
                print "Binning with mh_biggest = False"
        for n in range(len(peak_list)):

            p = peak_list[n]
            if self.verbose and n%500==0:
                print "Assigning possible transformations %d/%d" % (n, len(peak_list))
                sys.stdout.flush()

            for c in self.clusters:
                if p is c.mHPeak:
                    continue
                if self.mh_biggest:
                    if p.intensity > c.mHPeak.intensity:
                        continue
                t = self.check(p,c)
                if not t == None:
                    poss = Possible(c,t,t.transform(p),p.rt)
                    self.possible[p].append(poss)
                    self.clus_poss[c].append(poss)

        for n in range(len(peak_list)):
            p = peak_list[n]
            if len(self.possible[p])>1:
                self.todo.append(p)
            self.peak_idx[p] = n

        if self.corr_mat is not None:
            n_peaks = self.corr_mat.shape[0]
            assert n_peaks == len(peak_list)
            self.in_like, self.out_like = self._create_like_mats()

        if self.verbose:
            print "{} peaks to be re-sampled in stage 1".format(len(self.todo))

    def reset_counts(self):
        for p in self.peaks:
            for poss in self.possible[p]:
                poss.count = 0
        self.nSamples = 0

    def multi_sample(self, S):

        for s in range(S):
            self.do_gibbs_sample()

        # Fix the counts for things that don't get re-sampled
        for p in self.peaks:
            if p not in self.todo:
                self.possible[p][0].count += S
                self.possible[p][0].cluster.mu_mass = p.mass
                self.possible[p][0].cluster.mu_rt = p.rt

    def _create_like_mats(self):

        print "Creating likelihood matrices for shape clustering"
        n_peaks = (self.corr_mat.shape)[0]
        in_like = np.zeros((n_peaks, n_peaks))
        out_like = np.zeros((n_peaks, n_peaks))
        todo_indices = set([self.peak_idx[x] for x in self.todo])

        for n in np.arange(n_peaks-1):
            if n % 500 == 0:
                print n
            for m in np.arange(n+1, n_peaks):
                if self.corr_mat[n,m] != 0:
                    in_val = np.log(self.in_prob) + self._log_beta_pdf(self.corr_mat[n,m], self.in_alpha, self.in_beta)
                    out_val = np.log(self.out_prob) + self._log_beta_pdf(self.corr_mat[n,m], self.out_alpha, self.out_beta)
                else:
                    in_val = np.log(1-self.in_prob)
                    out_val = np.log(1-self.out_prob)
                in_like[n][m] = in_val
                in_like[m][n] = in_val
                out_like[n][m] = out_val
                out_like[m][n] = out_val
        return in_like, out_like

    def _log_beta_pdf(self, x, a, b):
        o = gammaln(a + b) - gammaln(a) - gammaln(b)
        o = o + (a-1)*np.log(x) + (b-1)*np.log(1-x)
        return o

    def do_gibbs_sample(self):

        base_like = self.out_like.sum(axis=0)
        for p in self.todo:

            # Remove from current cluster
            old_poss = self.Z[p]
            old_poss.cluster.N -= 1
            old_poss.cluster.members.remove(p)
            old_poss.cluster.rt_sum -= p.rt
            old_poss.cluster.mass_sum -= old_poss.transformed_mass
            old_trans = old_poss.transformation.name
            old_poss.cluster.peak_trans.remove(old_trans)

            post_max = -1e6
            post = []
            for poss in self.possible[p]:
                new_trans = poss.transformation.name
                if new_trans in poss.cluster.peak_trans:
                    # this transformation must not already exist in this cluster
                    new_post = float('-inf')
                else:
                    # prior
                    new_post = np.log(poss.cluster.N + (1.0*self.alpha)/(1.0*self.K))
                    # mass likelihood
                    new_post += poss.cluster.compute_mass_like(poss.transformation.transform(p))
                    # rt likelihood
                    new_post += poss.cluster.compute_rt_like(p.rt)
                    # peak shape likelihood
                    if self.corr_mat is not None:
                        this_peak_idx = self.peak_idx[p]
                        # excluding the current peak being sampled
                        in_pos = np.array([self.peak_idx[x] for x in poss.cluster.members])
                        if len(in_pos) > 0:
                            peak_shape_log_like = base_like[this_peak_idx] - self.out_like[this_peak_idx, in_pos].sum()
                            peak_shape_log_like += self.in_like[this_peak_idx, in_pos].sum()
                        else:
                            peak_shape_log_like = base_like[this_peak_idx]
                        new_post += peak_shape_log_like

                post.append(new_post)
                if new_post > post_max:
                    post_max = new_post

            assert len(post) == len(self.possible[p]), 'len(post)=%d, len(self.possible[p])=%d'
            post = np.array(post)
            post = np.exp(post - post_max)
            post /= post.sum()
            post = post.cumsum()
            pos = np.where(np.random.rand()<post)[0][0]
            new_poss = self.possible[p][pos]

            self.Z[p] = new_poss
            new_poss.cluster.N += 1
            new_poss.cluster.members.append(p)
            new_poss.cluster.rt_sum += p.rt
            new_poss.cluster.mass_sum += new_poss.transformed_mass
            new_trans = new_poss.transformation.name
            new_poss.cluster.peak_trans.append(new_trans)
            new_poss.count += 1

        self.nSamples += 1
        if self.nSamples % 100 == 0:
            print 'Sample %d' % self.nSamples

    def compute_posterior_probs(self):
        for p in self.peaks:
            for poss in self.possible[p]:
                poss.prob = (1.0*poss.count)/(1.0*self.nSamples)

    def display_probs(self):
        # Only displays the ones in todo
        for p in self.todo:
            print "Peak: {},{}".format(p.mass,p.rt)
            for poss in self.possible[p]:
                print "\t Cluster {}: {} ({} = {}), prob = {}".format(poss.cluster.id,poss.cluster.M,
                    poss.transformation.name,poss.transformed_mass,poss.prob)

    def check(self,peak,cluster):
        # Check RT first
        if np.abs(peak.rt - cluster.mHPeak.rt) > self.rt_tol:
            return None
        else:
            for t in self.transformations:
                tm = t.transform(peak)
                if np.abs((tm - cluster.M)/cluster.M)*1e6 < self.mass_tol:
                    return t
            return None

    def map_assign(self):
        # Assigns all peaks to their most likely cluster
        for c in self.clusters:
            c.N = 0
            c.rt_sum = 0
            c.mass_sum = 0
            c.peak_trans = []
        for p in self.peaks:
            possible_clusters = self.possible[p]
            # self.Z[p].cluster.N -= 1
            # self.Z[p].cluster.rt_sum -= p.rt
            # self.Z[p].cluster.mass_sum -= self.Z[p].transformed_mass
            if len(possible_clusters) == 1:
                self.Z[p] = possible_clusters[0]
            else:
                max_prob = 0.0
                for poss in possible_clusters:
                    if poss.prob > max_prob:
                        self.Z[p] = poss
                        max_prob = poss.prob
            self.Z[p].cluster.N += 1
            self.Z[p].cluster.rt_sum += p.rt
            self.Z[p].cluster.mass_sum += self.Z[p].transformed_mass
            new_trans = self.Z[p].transformation.name
            self.Z[p].cluster.peak_trans.append(new_trans)

    def cluster_plot(self, cluster):
        # Find the members and possible objects
        members = []
        possibles = []
        trans = []
        for p in self.peaks:
            if self.Z[p].cluster is cluster:
                members.append(p)
                possibles.append(self.Z[p])
                trans.append(self.Z[p].transformation)

        print "CLUSTER {}".format(cluster.id)
        max_intensity = 0
        for p in range(len(members)):
            print "Peak {} : {},{},{} -> {},{} (p={})".format(members[p]._get_key(), members[p].mass,members[p].rt,members[p].intensity,
                                                possibles[p].transformation.name,
                                                possibles[p].transformed_mass,
                                                possibles[p].prob)
            if members[p].intensity > max_intensity:
                max_intensity = members[p].intensity

        plt.figure()
        for p_ind in range(len(members)):
            p = members[p_ind]
            po = possibles[p_ind]
            plt.plot((p.mass,p.mass),(0,p.intensity/max_intensity),'k-')
            plt.annotate(po.transformation.name,(p.mass,p.intensity/max_intensity),
                        (p.mass,0.1+p.intensity/max_intensity),
                        arrowprops=dict(arrowstyle='->'),
                        textcoords='data')

        trans_names = [i.name for i in trans]
        all_trans_names = [i.name for i in self.transformations]
        for this_name in all_trans_names:
            r = (this_name,this_name + " [C13]")
            if r[0] in trans_names and r[1] in trans_names:
                pos_0 = trans_names.index(r[0])
                pos_1 = trans_names.index(r[1])
                print "{}/{} = {}".format(r[0],r[1],members[pos_0].intensity/members[pos_1].intensity)