# -*- coding: utf-8 -*-

"""extrae folding diff generator."""

import sys
import os
import glob

import perftask

class ProfilerExtraeFolddiffTask(perftask.TaskFrame):

    def __init__(self, parent, url, argv):

        # accept two argument to compare

        self.add_data_argument("folddir", metavar="path", nargs=2, help="folding directories to compare.")
        self.add_data_argument("region", metavar="name", nargs="*", help="region name(s) to compare.")
        #self.add_data_argument("region", metavar="name", nargs=1, help="region name to compare.")
        self.add_data_argument("-a", "--absolute-limit", metavar="value", type=float, help="mean value threshold of absolute event counts.")
        self.add_data_argument("-i", "--perins-limit", metavar="value", type=float, help="mean value threshold of per_instruction event counts.")

        try:
            import numpy as np 
            import pandas as pd 
            self.env['np']  = np
            self.env['pd']  = pd
            pd.options.mode.use_inf_as_na = True

        except Exception as e:
            self.parent.error_exit('ERROR: %s'%str(e))

    def perform(self):

        if not self.targs.region:
            self.parent.error_exit("Missing regions.")

        if len(self.targs.region) == 1:
            regions = self.targs.region*2
        else:
            regions = self.targs.region[:2]
 
        np = self.env['np']
        pd = self.env['pd']

        # collect data
        data = {0: {}, 1: {}, "meta": {}}
        diffsums = {}
        diffsums_ins = {}

        for idx, folddir in enumerate(self.targs.folddir):

            # folddir
            if not os.path.isdir(folddir):
                self.parent.error_exit('First argument is not a directory.: %s'%folddir)
            data[idx]['folddir'] = folddir

            # slope.csv
            flist = glob.glob(os.path.join(folddir, "*.slope.csv"))
            if len(flist) != 1:
                self.parent.error_exit("Folding directory, '%s', seems not valid."%folddir)
            csvfile = flist[0]
            data[idx]['csvfile'] = csvfile

            col_names = ["region", "group", "hwc", "ts", "event0", "event1"] 
            dtypes = {"region":str, "group":str, "hwc":str, "ts":np.float64, "event0":np.float64, "event1":np.float64}

            df = pd.read_csv(csvfile, sep=";", header=None, names=col_names, dtype=dtypes,
                skiprows=0, skipinitialspace=True)
            df = df.loc[df['region']==regions[idx]]

            data[idx]['csv'] = df[~df.hwc.str.endswith('per_ins')]
            data[idx]['csv_ins'] = df[df.hwc.str.endswith('per_ins')]

            hwcs = data[idx]['csv'].loc[data[idx]['csv']['region']==regions[idx]][["hwc", "event0"]]
            diffsums[idx] = hwcs.groupby("hwc").sum(skipna=True)

            hwcs_ins = data[idx]['csv_ins'].loc[data[idx]['csv_ins']['region']==regions[idx]][["hwc", "event0"]]
            diffsums_ins[idx] = hwcs_ins.groupby("hwc").sum(skipna=True)

            # kernel name
            csvbase = os.path.basename(flist[0])
            kernelname = csvbase.split(".codeblocks.")[0]
            data[idx]['kernelname'] = kernelname

            # fused.stats
            statfile = os.path.join(folddir, "%s.codeblocks.fused.stats"%kernelname)
            if not os.path.isfile(statfile):
                self.parent.error_exit("stat file, '%s', does not exist."%statfile)
            data[idx]['statfile'] = statfile

            col_names = ["group", "totinsnum", "totsamnum", "totmed", "totmad", "sigtimefac",
            #col_names = ["region", "group", "totinsnum", "totsamnum", "totmed", "totmad", "sigtimefac",
                "selintmin", "selintmax", "selinsnum", "selinsprop", "selinsmed", "selinsmad"] 
            dtypes = {"group":str, "totinsnum":np.int32, "totsamnum":np.int32, "totmed":np.float64,
            #dtypes = {"region":str, "group":str, "totinsnum":np.int32, "totsamnum":np.int32, "totmed":np.float64,
                "totmad":np.float64, "sigtimefac":np.float64, "selintmin":np.float64, "selintmax":np.float64,
                "selinsnum":np.int32, "selinsprop":np.int32, "selinsmed":np.float64, "selinsmad":np.float64}

            data[idx]['statdf'] = pd.read_csv(statfile, header=None, names=col_names, dtype=dtypes,
                skiprows=1, skipinitialspace=True, index_col=0)

        # construct regions

        if not regions[0] in data[0]['statdf'].index:
            self.parent.error_exit("Region, '%s', does not exist in both folding regions.."%regions[0])
        if not regions[1] in data[1]['statdf'].index:
            self.parent.error_exit("Region, '%s', does not exist in both folding regions.."%regions[1])


        est0 = data[0]['statdf'].loc[regions[0]].totmed*1.E-3
        est1 = data[1]['statdf'].loc[regions[1]].totmed*1.E-3
 
        if self.targs.absolute_limit:
            diffsums_limit = self.targs.absolute_limit*1000.
        else:
            # MHz
            if "PAPI_TOT_CYC" in diffsums[0].index:
                cpu_clock = diffsums[0].loc['PAPI_TOT_CYC'].event0/1000.
            else:
                cpu_clock = 2000.
            diffsums_limit = cpu_clock*100.

        index0 = set(diffsums[0].index[diffsums[0].event0>diffsums_limit].tolist())
        index1 = set(diffsums[1].index[diffsums[1].event0>diffsums_limit].tolist())
        index = list(index0 | index1)

        #diffsums[0] = diffsums[0].loc[diffsums[0].event0>diffsums_limit].loc[diffsums[1].event0>diffsums_limit]
        #diffsums[1] = diffsums[1].loc[diffsums[1].event0>diffsums_limit].loc[diffsums[0].event0>diffsums_limit]
        diffsums[0] = diffsums[0].loc[index]
        diffsums[1] = diffsums[1].loc[index]
        sorted_hwcs = (diffsums[1]/diffsums[0]*(est1/est0)).sort_values("event0", ascending=False, na_position='last')

        if self.targs.perins_limit:
            diffsums_ins_limit = self.targs.perins_limit*1000. 
        else:
            diffsums_ins_limit = 100. 

        index_ins0 = set(diffsums_ins[0].index[diffsums_ins[0].event0>diffsums_ins_limit].tolist())
        index_ins1 = set(diffsums_ins[1].index[diffsums_ins[1].event0>diffsums_ins_limit].tolist())
        index_ins = list(index_ins0 | index_ins1)
        diffsums_ins[0] = diffsums_ins[0].loc[index_ins]
        diffsums_ins[1] = diffsums_ins[1].loc[index_ins]
        sorted_hwcs_ins = (diffsums_ins[1]/diffsums_ins[0]).sort_values("event0", ascending=False, na_position='last')

        data["meta"]["regionnames"] = regions
        data["meta"]["regionshortnames"] = [ region.split("_mp_")[1] for region in regions ]
        data["meta"]["diffsorted"] = sorted_hwcs
        data["meta"]["diffsorted_ins"] = sorted_hwcs_ins

        return 0, {"D": data}

