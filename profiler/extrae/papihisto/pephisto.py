# -*- coding: utf-8 -*-

"""extrae prv parser task module."""

import sys
import os
import numpy as np
import perftask

try:
    from  matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf = PdfPages('papihisto.pdf')
except Exception as e:
    print ('ERROR: matplotlib module is not loaded: %s'%str(e))
    sys.exit(-1)

TITLE_SIZE = 20
SUBTITLE_SIZE = 16
TEXT_SIZE = 14
LABEL_SIZE = 16
LINEWIDTH = 3

user_function_event = "60000019"
number_of_bins = 200 

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

        self.add_data_argument("tracefile", metavar="path", evaluate=False, help="Extrae trace prv file.")

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


        # generate plot pdf file

        for idx, (hwc_name, func_data_sets) in enumerate(hwc_bins.items()):


            keys = func_data_sets.keys()
            labels = []
            for  fid in keys:
                desc = self.pcf.get_pcf_event(int(user_function_event))['values'][int(fid)]
                item = desc[-1].strip() if isinstance(desc, (list, tuple)) else desc.strip()
                if item.startswith("["): item = item[1:].lstrip()
                if item.endswith("]"): item = item[:-1].rstrip()
                labels.append(item.split("_mp_")[-1])

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
