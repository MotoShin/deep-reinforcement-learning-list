import numpy as np
import pandas as pd


class DataShaping(object):

    @staticmethod
    def makeCsv(lst, kind, file_name):
        csv_lst = []
        cols = ['episode']
        num_simulation = len(lst)
        num_episode = len(lst[0])

        for sim in range(num_simulation):
            cols.append("{}sim_{}".format(sim+1, kind))

        for epi in range(num_episode):
            one_line = [epi+1]
            for sim in range(num_simulation):
                one_line.append(lst[sim][epi])
            csv_lst.append(one_line)

        df = pd.DataFrame(csv_lst, columns=cols)
        df.to_csv('output/' + file_name, index=False)

    @staticmethod
    def makeParamCsv(lst, kind, file_name):
        csv_lst = []
        cols = ['update_num']
        num_simulation = len(lst)
        num_episode = len(lst[0])

        for sim in range(num_simulation):
            cols.append("{}sim_{}".format(sim+1, kind))

        for epi in range(num_episode):
            one_line = [epi+1]
            for sim in range(num_simulation):
                one_line.append(lst[sim][epi])
            csv_lst.append(one_line)

        df = pd.DataFrame(csv_lst, columns=cols)
        df.to_csv('output/' + file_name, index=False)
