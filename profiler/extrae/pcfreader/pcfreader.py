# -*- coding: utf-8 -*-

"""extrae pcf parser task module."""

import os
import perftask

class PcfReadTask(perftask.TaskFrame):

    # check path
    def __init__(self, parent, url, argv):

        self.add_data_argument("pcfdata", metavar="path", evaluate=False, help="Extrae trace pcf file.")

    def perform(self):

        if not os.path.isfile(self.targs.pcfdata):
            self.parent.error_exit("'{}' is not found.".format(self.targs.pcfdata))

        events = {}

        BLANK, HEAD, VALUE = 0, 1, 2
        state = BLANK

        with open(self.targs.pcfdata, 'r') as fh:

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
                            events[int(items[1])] = eventtype
                elif state == VALUE:
                    items = line.split()
                    if len(items) > 2:
                        eventvalues[int(items[0])] = (items[1], ' '.join(items[2:]))
                    elif len(items) == 2:
                        eventvalues[int(items[0])] = ' '.join(items[1:])

            for eventtype in eventtypes:
                eventtype['values'] = eventvalues

        forward = {"pcfdata": events}

        return 0, forward
