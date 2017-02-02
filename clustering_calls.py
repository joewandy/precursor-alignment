import operator
import os
import sys

import cPickle
import gzip
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from adduct_cluster import AdductCluster, BetaLike
from second_stage_clusterer import DpMixtureGibbs

def _run_first_stage_clustering(j, peak_data, hp, trans_filename, mh_biggest):

    sys.stdout.flush()
    peak_list = peak_data.features
    corr_mat = peak_data.corr_mat
    shape_clustering = True if corr_mat is not None else False

    ac_dir = '/Users/joewandy/Dropbox/Analysis/precursor/multibeers/notebooks/pickles/acs/'
    base_name, ext = os.path.splitext(peak_data.filename)
    file_name = '%s_masstol_%d_rttol_%d_mhbiggest_%s_shape_%s.ac' % (base_name, hp.within_file_mass_tol,
                                                                        hp.within_file_rt_tol, mh_biggest, shape_clustering)
    file_path = os.path.join(ac_dir, file_name)
    try:
        with gzip.GzipFile(file_path, 'rb') as f:
            ac = cPickle.load(f)
            print "Loaded ac from %s" % file_path
    except (TypeError, IOError, EOFError):

        ac = AdductCluster(mass_tol=hp.within_file_mass_tol, rt_tol=hp.within_file_rt_tol,
                           alpha=hp.alpha_mass, mh_biggest=mh_biggest, transformation_file=trans_filename, verbose=2,
                           corr_mat=corr_mat)
        ac.init_from_list(peak_list)

        with gzip.GzipFile(file_path, 'wb') as f:
            cPickle.dump(ac, f, protocol=cPickle.HIGHEST_PROTOCOL)
        print "Saved to %s" % file_path

    print 'Running Gibbs sampler on %s' % peak_data.filename
    print '- concentration param=%.2f' % ac.alpha
    print '- mass_tol=%.2f, rt_tol=%.2f' % (ac.mass_tol, ac.rt_tol)
    print ac.like_object
    ac.multi_sample(hp.mass_clustering_n_iterations)
    return ac

def _run_second_stage_clustering(n, cluster_list, hp, seed, verbose=False):

    if seed == -1:
        seed = 1234567890

    masses = []
    rts = []
    word_counts = []
    origins = []
    for cluster in cluster_list:
        masses.append(cluster.mu_mass)
        rts.append(cluster.mu_rt)
        word_counts.append(cluster.word_counts)
        origins.append(cluster.origin)
    data = (masses, rts, word_counts, origins)

    # run dp clustering for each top id
    dp = DpMixtureGibbs(data, hp, seed=seed, verbose=verbose)
    dp.nsamps = hp.rt_clustering_nsamps
    dp.burn_in = hp.rt_clustering_burnin
    dp.run()

    # read the clustering results back
    matching_results = []
    results = {}
    for matched_set in dp.matching_results:
        members = [cluster_list[a] for a in matched_set]
        memberstup = tuple(members)
        matching_results.append(memberstup)
        if matched_set in results:
            results[matched_set] += 1
        else:
            results[matched_set] = 1

    output = "n " + str(n) + "\tcluster_list=" + str(len(cluster_list)) + "\tlast_K = " + str(dp.last_K)
    if verbose:
        output += "\n"
        mass_list = dp.masses.tolist()
        rt_list = dp.rts.tolist()
        adduct_list = dp.word_counts_list
        for i in range(len(mass_list)):
            output += "#" + str(i) + "\n"
            output += "  mass = %.5f rt=%.3f\n" % (mass_list[i], rt_list[i])
            output += "  adducts = %s\n" % (adduct_list[i].tolist())
        output += "clustering results\n"
        sorted_results = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
        for key, value in sorted_results:
            output += "  %s = %d\n" % (key, value)
    print output
    sys.stdout.flush()

    return matching_results