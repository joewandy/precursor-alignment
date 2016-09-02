#!/usr/bin/env python

import os
import sys
basedir = '..'
sys.path.append(basedir)

from shared_bin_matching import SharedBinMatching as Aligner
from preprocessing import FileLoader
from models import AlignmentHyperPars

# --------------------------------------------------------------
# 1. load input files
# --------------------------------------------------------------

loader = FileLoader()
input_dir = '../input/beer3pos'
data_list = loader.load_model_input(input_dir, verbose=True)

# --------------------------------------------------------------
# 2. define parameters
# --------------------------------------------------------------

hp = AlignmentHyperPars()
hp.within_file_mass_tol = 3
hp.within_file_rt_tol = 10
hp.across_file_mass_tol = 6
hp.across_file_rt_tol = 15
hp.alpha_mass = 1
hp.mass_clustering_n_iterations = 1000
hp.dp_alpha = 1000.0
hp.beta = 0.1
hp.rt_clustering_nsamps = 1000
hp.rt_clustering_burnin = 500

# --------------------------------------------------------------
# 3. perform a direct-matching MW alignment
# --------------------------------------------------------------

transformation_file = None
aligner_mw = Aligner(data_list, transformation_file, hp)
match_mode = 0
aligner_mw.run(match_mode)

print 'Number of aligned peaksets', len(aligner_mw.alignment_results)
df = aligner_mw.print_peaksets()
df.to_csv('demo_mw.csv', index=False)

# --------------------------------------------------------------
# 4. perform alignment via matching the IP clusters (Cluster-Match)
# --------------------------------------------------------------

transformation_file = '../pos_transformations.yml'
aligner_cm = Aligner(data_list, transformation_file, hp)
match_mode = 1
aligner_cm.run(match_mode)

# export all aligned peaksets
df = aligner_cm.print_peaksets()
df.to_csv('demo_cluster_match_all.csv', index=False)

# export the aligned peaksets only for the non-singleton IP clusters
df = aligner_cm.print_peaksets(exclude_singleton=True)
df.to_csv('demo_cluster_match_non_singleton.csv', index=False)

# --------------------------------------------------------------
# 4. perform alignment via clustering the IP clusters (Cluster-Cluster)
# --------------------------------------------------------------

transformation_file = '../pos_transformations.yml'
aligner_cc = Aligner(data_list, transformation_file, hp, parallel=False)
match_mode = 2
aligner_cc.run(match_mode)

df = aligner_cc.print_peaksets(exclude_singleton=True)
df.to_csv('demo_cluster_cluster_all.csv', index=False)