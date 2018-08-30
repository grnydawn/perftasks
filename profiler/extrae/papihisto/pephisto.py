# -*- coding: utf-8 -*-

"""extrae prv parser task module."""

import sys
import os
import numpy as np
import perftask

try:
    from  matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages('papihosto.pdf')
except Exception as e:
    print ('ERROR: matplotlib module is not loaded: %s'%str(e))
    sys.exit(-1)

TITLE_SIZE = 20
SUBTITLE_SIZE = 16
TEXT_SIZE = 14
LABEL_SIZE = 16
LINEWIDTH = 3

user_function_event = "60000019"

#
#import os
#import re
#import gzip
#import perftask
#import StringIO
#
#_re_cesm_stat1 = re.compile(r"\s*Total\sranks\sin\scommunicator")
#_re_cesm_stat2 = re.compile(r"\s*\*+\sGLOBAL\sSTATISTICS\s+\(")
#_re_stat_table1 = re.compile(r"name\s+ncalls\s+nranks\s+mean_time")
#_re_stat_table2 = re.compile(r"name\s+processes\s+threads\s+count")
#
class ExtraePcfData(object):

    def __init__(self, path):

        self.events = {}

        BLANK, HEAD, VALUE = 0, 1, 2
        state = BLANK

        with open(path, 'r') as fh:

            eventtypes = []
            eventvalues = {}

            for line in fh:

                line = line.strip()
                if len(line) == 0:
                    state = BLANK
                    continue

                if state == BLANK:
                    if line == "EVENT_TYPE":
                        for eventtype in eventtypes:
                            eventtype['values'] = eventvalues
                        eventtypes = []
                        eventvalues = {}
                        state = HEAD
                elif state == HEAD:
                    if line == "VALUES":
                        state = VALUE
                    else:
                        items = line.split()
                        if len(items) > 2:
                            eventtype = {'desc': ' '.join(items[2:])}
                            eventtypes.append(eventtype)
                            self.events[int(items[1])] = eventtype
                elif state == VALUE:
                    items = line.split()
                    if len(items) > 2:
                        eventvalues[int(items[0])] = (items[1], ' '.join(items[2:]))
                    elif len(items) == 2:
                        eventvalues[int(items[0])] = ' '.join(items[1:])

            for eventtype in eventtypes:
                eventtype['values'] = eventvalues

    def get_pcf_events(self, eventrange):
        for eventype, (desc, values) in self.events.items():
            if eventype in eventrange:
                yield desc, values

    def get_pcf_event(self, eventtype):
        return self.events[eventtype]

class ProfilerExtraePapiHistoTask(perftask.TaskFrame):

    # check path
    def __init__(self, parent, url, argv):

        self.set_data_argument("tracefile", metavar="path", evaluate=False, help="Extrae trace prv file.")
#
#        try:
#            import pandas
#            self.env["pandas"] = self.env["pd"] = pandas
#        except ImportError as err:
#            self.error_exit("pandas module is not found.")

    def perform(self):

        # load prv trace line by line
        assert os.path.isfile(self.targs.tracefile), "Trace file does not exist."
        root, ext = os.path.splitext(self.targs.tracefile)
        metafile = root+".pcf"
        assert os.path.isfile(metafile), "Trace meta file, '%s', does not exist."%metafile

        self.pcf = ExtraePcfData(metafile)

        hwc_bins = {} # { hwc: { fid: []} }
        buf = {}

        # construct histogram
        with open(self.targs.tracefile) as f:
            for line in f:
                if line and line[0] == "2":
                    items = line.rstrip().split(":")
                    if user_function_event in items:
                        idx = items.index(user_function_event)
                        ts = int(items[5])
                        key = ":".join(items[1:3])
                        if items[idx+1] == "0":
                            if key not in buf:
                                print("WARN: key, '%s'. is not in a buf. Discarded."%key)
                                #import pdb; pdb.set_trace()
                            else:
                                prev_ts, prev_fid, prev_events = buf[key].pop()
                                for event_id, event_value in zip(prev_events[0::2], prev_events[1::2]):
                                    if event_id.startswith("4200") and event_id in items:
                                        eidx = items.index(event_id)
                                        diff_val = int(items[eidx+1]) - int(event_value)
                                        diff_ts = ts - prev_ts
                                        value = float(diff_val) / float(diff_ts)
                                        if value > 0:
                                            if event_id not in hwc_bins:
                                                hwc_bin = {}
                                                hwc_bins[event_id] = hwc_bin
                                            hwc_bin = hwc_bins[event_id]
                                            if prev_fid not in hwc_bin:
                                                fid_list = []
                                                hwc_bin[prev_fid] = fid_list
                                            fid_list = hwc_bin[prev_fid]
                                            fid_list.append(value)
                        else:
                            if key in buf:
                                buf[key].append((ts, items[idx+1], items[idx+2:]))
                            else:
                                buf[key] = [(ts, items[idx+1], items[idx+2:])]

                    # format: <linetype>:<mpi rank><thread id>:<mpi rank><thread id>:<timestamp>:<event type>:<event value>:...
                    # hwc events: 4200XXXX
                    # user function event: 60000019

        #import pdb; pdb.set_trace()                            
        #np.random.seed(19680801)
        number_of_bins = 100 

        # An example of three data sets to compare
#        number_of_data_points = 387
#        func_names = ["A", "B", "C"]
#        data1 = [np.random.normal(0, 1, number_of_data_points),
#                     np.random.normal(6, 1, number_of_data_points),
#                     np.random.normal(-3, 1, number_of_data_points)]
#        data2 = [np.random.normal(0, 1, number_of_data_points),
#                     np.random.normal(6, 1, number_of_data_points),
#                     np.random.normal(-3, 1, number_of_data_points)]

        # generate plot pdf file

        for idx, (hwc_name, func_data_sets) in enumerate(hwc_bins.items()):


            keys = func_data_sets.keys()
            labels = []
            for  fid in keys:
                desc = self.pcf.get_pcf_event(int(user_function_event))['values'][int(fid)]
                if isinstance(desc, (list, tuple)):
                    labels.append(desc[-1])
                else:
                    labels.append(desc)

            #labels = func_data_sets.keys()
            data_sets = [np.asarray(func_data_sets[l]) for l in keys]
            hist_max = np.max([np.max(d) for d in data_sets])
            hist_min = np.min([np.min(d) for d in data_sets])

            #import pdb; pdb.set_trace()
            # Computed quantities to aid plotting
            #hist_range = (np.min(data_sets), np.max(data_sets))
            hist_range = (hist_min, hist_max)
            binned_data_sets = [
                np.histogram(d, range=hist_range, bins=number_of_bins)[0]
                for d in data_sets
            ]
            binned_maximums = np.max(binned_data_sets, axis=1)
            x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))

            # The bin_edges are the same for all of the histograms
            bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
            centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
            heights = np.diff(bin_edges)

            # Cycle through and plot each histogram
            fig, ax = plt.subplots(figsize=(8, 6))

            for x_loc, binned_data in zip(x_locations, binned_data_sets):
                lefts = x_loc - 0.5 * binned_data
                ax.barh(centers, binned_data, height=heights, left=lefts)

            ax.set_title(self.pcf.get_pcf_event(int(hwc_name))['desc'])
            ax.set_xticks(x_locations)
            ax.set_xticklabels(labels)

            ax.set_ylabel("Event ratio (events / nano sec.)")
            ax.set_xlabel("functions")

            pdf.savefig()
            plt.cla()

        pdf.close()



#        def cesm_stat(c):
#            if _re_cesm_stat1.match(c):
#                return True
#            elif _re_cesm_stat2.match(c):
#                return True
#
#        contents = {}
#
#        # read timing file
#        for path in self.targs.timingfile:
#            path = os.path.abspath(os.path.realpath(path))
#            try:
#                with gzip.open(path) as zf:
#                    contents[path] = zf.read()
#            except IOError as err:
#                with open(path) as f:
#                    contents[path] = f.read()
#
#        # handle other options
#
#        # default action
#
#        for path, content in contents.items():
#            if cesm_stat(content):
#                self.env["D"].append(self._cesm_stat(content))
#            else:
#                self.error_exit("Unknown cesm timing file: %s"%path)
#
#        if not self.env["D"]:
#            self.error_exit("No cesm timing input is found.")
#
#        self.add_forward("D", self.env["D"])
#   
#        return 0     
#    
#    def _cesm_stat(self, content):
#        def _read_table(c, start):
#            table_str = c[start:].replace("(", "").replace(")", "").strip()
#            return self.env["pd"].read_csv(StringIO.StringIO(table_str), sep="\s+",
#                error_bad_lines=False, index_col=0) 
#
#        stat_df = None
#        match = _re_stat_table1.search(content)
#        if match:
#            stat_df = _read_table(content, match.start())
#        else:
#            match = _re_stat_table2.search(content)
#            if match:
#                stat_df = _read_table(content, match.start())
#        return stat_df
