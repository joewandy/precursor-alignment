from collections import namedtuple
import csv
import os
import numpy as np

import utils

class AlignmentHyperPars(object):

    def __init__(self):

        self.within_file_mass_tol = 3.0
        self.within_file_rt_tol = 10.0
        self.across_file_mass_tol = 5.0
        self.across_file_rt_tol = 30.0

        self.alpha_mass = 1.0
        self.dp_alpha = 1000.0
        self.beta = 0.1

        self.mass_clustering_n_iterations = 200
        self.rt_clustering_nsamps = 100
        self.rt_clustering_burnin = 0

        self.matching_alpha = 0.3
        self.second_stage_clustering_use_mass_likelihood = True
        self.second_stage_clustering_use_rt_likelihood = True
        self.second_stage_clustering_use_adduct_likelihood = True


    def __repr__(self):
        return "Hyperparameters " + utils.print_all_attributes(self)

class ClusteringHyperPars(object):

    def __init__(self):
        self.rt_prec = 10
        self.mass_prec = 100
        self.rt_prior_prec = 100
        self.mass_prior_prec = 100
        self.alpha = float(100)

    def __repr__(self):
        return "Hyperparameters " + utils.print_all_attributes(self)

class Feature(object):

    def __init__(self, peak_id, mass, rt, intensity, parent_file, fingerprint=None):
        self.peak_id = int(peak_id)
        self.mass = mass
        self.intensity = intensity
        self.rt = rt
        self.parent_file = parent_file
        self.fingerprint = fingerprint
        self.gt_metabolite = None   # used for synthetic data
        self.gt_adduct = None       # used for synthetic data

    def _get_key(self):
        return (self.peak_id, self.parent_file)

    def __eq__(self, other):
        return self._get_key() == other._get_key()

    def __hash__(self):
        return hash(self._get_key())

    def __repr__(self):
        return "peak_id " + str(self.peak_id) + " parent_file " + str(self.parent_file) + \
            " mass " + str(self.mass) + " rt " + str(self.rt) + " intensity " + str(self.intensity)

# Transformation = namedtuple('Transformation', ['trans_id', 'name', 'sub', 'mul', 'iso'])
# DiscreteInfo = namedtuple('DiscreteInfo', ['possible', 'transformed', 'matRT', 'bins', 'prior_masses', 'prior_rts'])

class PeakData(object):

    def __init__(self, features, filename, corr_mat=None):

        # list of feature, database entry and transformation objects
        self.features = features
        self.num_peaks = len(features)
        self.corr_mat = corr_mat
        self.filename = filename

        # the same data as numpy arrays for convenience
        self.mass = np.array([f.mass for f in self.features])[:, None]              # N x 1
        self.rt = np.array([f.rt for f in self.features])[:, None]                  # N x 1
        self.intensity = np.array([f.intensity for f in self.features])[:, None]    # N x 1

class DatabaseEntry(object):

    # this is mostly called by io.FileLoader.load_database()
    def __init__(self, db_id, name, formula, mass, rt):
        self.db_id = db_id
        self.name = name
        self.formula = formula
        self.mass = mass
        self.rt = rt

    def set_ranges(self, mass_tol):
        self.mass_range = utils.mass_range(self.mass, mass_tol)

    def get_begin(self):
        return self.mass_range[0]

    def get_end(self):
        return self.mass_range[1]

    def __key(self):
        return (self.db_id, self.name)

    def __eq__(self, x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self):
        return "DatabaseEntry " + utils.print_all_attributes(self)

class AlignmentRow(object):

    def __init__(self, new_row_id):
        self.row_id = new_row_id
        self.features = []
        self.aligned = False
        self.grouped = False

    def __repr__(self):
        features_str = ' '.join([str(f) for f in self.features])
        return "row_id " + str(self.row_id) + " aligned " + str(self.aligned) + " grouped " + str(self.grouped) + \
            " features [" + features_str + "]"

    def add_features(self, another_row):
        '''Adds the features from another row into this row'''
        self.features.extend(another_row.features)

    def get_average_mass(self):
        '''Computes the average mass'''
        total_mass = 0
        for feature in self.features:
            total_mass = total_mass + feature.mass
        return total_mass / len(self.features)

    def get_mass_range(self, dmz, absolute_mass_tolerance=True):
        '''Computes tolerance window for mass difference'''
        average_mass = self.get_average_mass()
        if absolute_mass_tolerance:
            mass_lower = average_mass - dmz
            mass_upper = average_mass + dmz
        else:
            mass_centre = average_mass
            interval = mass_centre * dmz * 1e-6
            mass_lower = mass_centre - interval
            mass_upper = mass_centre + interval
        return (mass_lower, mass_upper)

    def get_average_rt(self):
        '''Computes the average RT'''
        total_rt = 0
        for feature in self.features:
            total_rt = total_rt + feature.rt
        return total_rt / len(self.features)

    def get_rt_range(self, drt):
        '''Computes tolerance window for rt difference'''
        average_rt = self.get_average_rt()
        rt_lower = average_rt - drt
        rt_upper = average_rt + drt
        return (rt_lower, rt_upper)

    def get_average_intensity(self):
        '''Computes the average intensity'''
        total_intensity = 0
        for feature in self.features:
            total_intensity = total_intensity + feature.intensity
        return total_intensity / len(self.features)

    def get_average_fingerprint(self):
        '''Computes the average fingerprint'''
        total_fingerprint = np.zeros_like(self.features[0].fingerprint)
        for feature in self.features:
            total_fingerprint = total_fingerprint + feature.fingerprint
        return total_fingerprint / len(self.features)

    def is_within_tolerance(self, another_row, dmz, drt, absolute_mass_tolerance=True):
        if another_row.aligned == True:
            return False
        else:
            # only process unaligned rows
            if dmz > 0:
                mass_to_check = another_row.get_average_mass()
                mass_lower, mass_upper = self.get_mass_range(dmz, absolute_mass_tolerance=absolute_mass_tolerance)
                if mass_lower < mass_to_check < mass_upper:
                    if drt > 0:
                        # need to check rt as well
                        rt_to_check = another_row.get_average_rt()
                        rt_lower, rt_upper = self.get_rt_range(drt)
                        if rt_lower < rt_to_check < rt_upper:
                            return True # both mass and rt okay
                        else:
                            return False # mass okay but rt not okay
                    else:
                        # no need to check rt
                        return True # mass okay
                else:
                    return False # mass not okay
            else:
                # compare by rt only
                if drt > 0:
                    rt_to_check = another_row.get_average_rt()
                    rt_lower, rt_upper = self.get_rt_range(drt)
                    if rt_lower < rt_to_check < rt_upper:
                        return True # rt okay
                    else:
                        return False # rt not okay
                else:
                    return True # nothing to check. shoudln't happen

    def pick_nearest(self, other_rows):
        all_diff = []
        for other in other_rows:
            diff = abs(self.get_average_mass() - other.get_average_mass())
            all_diff.append(diff)
        min_index = all_diff.index(min(all_diff))
        return other_rows[min_index]

    def get_begin(self):
        '''For interval tree'''
        return int(self.get_average_mass())

    def get_end(self):
        '''For interval tree'''
        return int(self.get_average_mass())

class AlignmentFile(object):

    def __init__(self, full_path, verbose):
        self.full_path = full_path
        parent_dir, filename = os.path.split(full_path)
        self.parent_dir = parent_dir
        self.filename = filename
        self.rows = []
        self.verbose = verbose

    def __repr__(self):
        return "full_path " + self.full_path + " with " + str(len(self.rows)) + " rows"

    def get_data(self):
        '''Reads features from input file pointed by full_path'''
        row_id = 0
        peak_id = 0
        if self.verbose:
            print "Loading " + self.full_path
        with open(self.full_path, 'rb') as f:
            # skip header if present
            has_header = csv.Sniffer().has_header(f.read(1024))
            f.seek(0)
            reader = csv.reader(f, delimiter='\t')
            if has_header:
                next(reader)  # skip header row
            # read the file contents
            for file_row in reader:
                # initialise feature
                mass = file_row[0]
                charge = file_row[1]
                intensity = file_row[2]
                rt = file_row[3]
                feat = Feature(peak_id, mass, charge, intensity, rt, self)
                peak_id = peak_id + 1
                # initialise row
                alignment_row = AlignmentRow(row_id)
                alignment_row.features.append(feat)
                row_id = row_id + 1
                self.rows.append(alignment_row)
        # print summary
        row_count = str(len(self.rows))
        if self.verbose:
            print " - " + row_count + " rows read"

    def add_rows(self, other_rows):
        '''Adds the rows from another file into this file'''
        self.rows.extend(other_rows)

    def add_row(self, another_row):
        '''Adds a row from another file into this file'''
        self.rows.append(another_row)

    def get_candidate_rows(self, reference_row, dmz, drt):
        candidates = []
        for candidate in self.rows:
            if candidate.is_within_tolerance(reference_row, dmz, drt):
                candidates.append(candidate)
        return candidates

    def reset_aligned_status(self):
        for row in self.rows:
            row.aligned = False

    def get_unaligned_rows(self):
        unaligned = []
        for row in self.rows:
            if row.aligned == False:
                unaligned.append(row)
        return unaligned

    def get_ungrouped_rows(self, reference_row, drt):
        candidates = []
        for candidate in self.rows:
            if candidate.grouped == True:
                continue
            else:
                if candidate.is_within_tolerance(reference_row, -1, drt):
                    candidates.append(candidate)
        return candidates

    def get_all_features(self):
        features = []
        for row in self.rows:
            features.extend(row.features)
        return features
