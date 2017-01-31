import csv
import glob
import os
import sys

import scipy.io as sio
import scipy

from models import PeakData, Feature, DatabaseEntry
import utils

class FileLoader:

    def load_model_input(self, input_file, corr_rt_window, verbose=False):
        """ Load everything that a clustering model requires """

        # if this is a directory, process all files inside
        # input_file = os.path.abspath(input_file)
        # print os.path.isdir(input_file)

        if os.path.isdir(input_file):

            print 'Loading files from', input_file

            # find all the .txt and csv files in input_dir
            input_dir = input_file
            filelist = []
            types = ('*.csv', '*.txt')
            starting_dir = os.getcwd() # save the initial dir to restore
            os.chdir(input_dir)
            for files in types:
                filelist.extend(glob.glob(files))
            filelist = utils.natural_sort(filelist)
            self.file_list = filelist

            # load the files
            file_id = 0
            data_list = []
            all_features = []
            for file_path in filelist:
                full_path = os.path.abspath(file_path)
                features, corr_adjacency = self.load_features(full_path, file_id, corr_rt_window)
                file_id += 1
                data = PeakData(features, file_path, corr_mat=corr_adjacency)
                all_features.extend(features)
                data_list.append(data)
                sys.stdout.flush()
            os.chdir(starting_dir)
            return data_list

        else:

            print input_file, 'must be a directory containing the input file'

    def load_features(self, input_file, file_id, corr_rt_window):

        # first load the features
        features = []
        if input_file.endswith(".csv"):
            features = self.load_features_csv(input_file, file_id)
        elif input_file.endswith(".txt"):
            # in SIMA (.txt) format
            features = self.load_features_sima(input_file, file_id)
        print str(len(features)) + " features read from " + input_file

        # also check if the correlation matrix is there, if yes load it too
        corr_mat = None
        front_part, extension = os.path.splitext(input_file)
        matfile = '%s_rtwindow_%d.corr.mat' % (front_part, corr_rt_window)
        if os.path.isfile(matfile):
            print "Reading peak shape correlations from " + matfile
            mdict = sio.loadmat(matfile)
            corr_mat = mdict['corr_mat']

            adjacency = {}
            for p in features:
                adjacency[p] = {}

            cx = scipy.sparse.coo_matrix(corr_mat)
            for i,j,v in zip(cx.row, cx.col, cx.data):
                if not i == j:
                    if v==1:
                        v = 0.99
                    elif v==0:
                        v = 0.01
                    adjacency[features[i]][features[j]] = v

        return features, adjacency

    def detect_delimiter(self, input_file):
        with open(input_file, 'rb') as csvfile:
            header = csvfile.readline()
            if header.find(":")!=-1:
                return ':'
            elif header.find(",")!=-1:
                return ','

    def load_features_csv(self, input_file, file_id):
        features = []
        if not os.path.exists(input_file):
            return features
        delim = self.detect_delimiter(input_file)
        # print 'delim is %s' % delim
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=delim)
            next(reader, None)  # skip the headers
            for elements in reader:
                if len(elements)==5:
                    feature_id = utils.num(elements[0])
                    mz = utils.num(elements[1])
                    rt = utils.num(elements[2])
                    intensity = utils.num(elements[3])
                    identification = elements[4] # unused
                    feature = Feature(feature_id, mz, rt, intensity, file_id)
                elif len(elements)==4:
                    feature_id = utils.num(elements[0])
                    mz = utils.num(elements[1])
                    rt = utils.num(elements[2])
                    intensity = utils.num(elements[3])
                    feature = Feature(feature_id, mz, rt, intensity, file_id)
                features.append(feature)
        return features

    def load_features_sima(self, input_file, file_id):
        features = []
        if not os.path.exists(input_file):
            return features
        with open(input_file, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            feature_id = 1
            for elements in reader:
                mass = float(elements[0])
                charge = float(elements[1])
                mass = mass/charge
                intensity = utils.num(elements[2])
                rt = utils.num(elements[3])
                feature = Feature(feature_id, mass, rt, intensity, file_id)
                if len(elements)>4:
                    # for debugging with synthetic data
                    gt_peak_id = utils.num(elements[4])
                    gt_metabolite_id = utils.num(elements[5])
                    gt_adduct_type = elements[6]
                    feature.gt_metabolite = gt_metabolite_id
                    feature.gt_adduct = gt_adduct_type
                features.append(feature)
                feature_id = feature_id + 1
        return features

    def load_database(self, database):
        moldb = []
        if not os.path.exists(database):
            return moldb
        with open(database, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for elements in reader:
                if len(elements)==5:
                    mol = DatabaseEntry(db_id=elements[0], name=elements[1], formula=elements[2], \
                                        mass=utils.num(elements[3]), rt=utils.num(elements[4]))
                    moldb.append(mol)
                elif len(elements)==4:
                    mol = DatabaseEntry(db_id=elements[0], name=elements[1], formula=elements[2], \
                                        mass=utils.num(elements[3]), rt=0)
                    moldb.append(mol)
        return moldb

def main(argv):

    loader = FileLoader()
    input_file = 'input/std1_csv_full_old'

    import time
    start = time.time()
    data_list = loader.load_model_input(input_file)
    print len(data_list)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("TIME TAKEN {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

if __name__ == "__main__":
   main(sys.argv[1:])

