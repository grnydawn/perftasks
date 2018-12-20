# -*- coding: utf-8 -*-

"""matplotlib task module."""

from __future__ import unicode_literals

import os
import perftask

# TODO: support matplotlib and pyplot functions

class PlotMPLTask(perftask.Task):
    """matplotlib utility task

        this task is ...
    """
    def __init__(self, parent, tid, path, fragment, argv):

        try:
            import matplotlib
            import matplotlib.pyplot
            self.env["matplotlib"] = self.env["mpl"] = matplotlib
            self.env["pyplot"] = self.env["plt"] = matplotlib.pyplot
        except ImportError as err:
            self.parent.error_exit(str(err))

        try:
            import numpy
            self.env["numpy"] = self.env["np"] = numpy
        except ImportError as err:
            pass

        try:
            import pandas
            self.env["pandas"] = self.env["pd"] = pandas
        except ImportError as err:
            pass

        self.add_data_argument('data', metavar='data', evaluate=True, nargs="*",
                               autoimport=True, help='input data for plotting.')

        self.add_option_argument('-f', metavar='figure creation',
                                 help='define a figure for plotting.')
        self.add_option_argument('-t', '--title', metavar='title',
                                 help='title  plotting.')
        self.add_option_argument('-p', '--plot', metavar='plot type',
                                 action='append',
                                 help='plot type for plotting.')
        self.add_option_argument('-s', '--save', metavar='save', action='append',
                                 help='file path to save png image.')
        self.add_option_argument('-x', '--xaxis', metavar='xaxis',
                                 action='append',
                                 help='axes function wrapper for x axis settings.')
        self.add_option_argument('-y', '--yaxis', metavar='yaxis', action='append',
                                 help='axes function wrapper for y axis settings.')
        self.add_option_argument('-z', '--zaxis', metavar='zaxis', action='append',
                                 help='axes function wrapper for z axis settings.')
        self.add_option_argument('-g', action='store_true',
                                 help='grid for ax plotting.')
        self.add_option_argument('-l', action='store_true',
                                 help='legend for ax plotting')
        self.add_option_argument('--pages', metavar='pages',
                                 help='page settings.')
        self.add_option_argument('--page-calc', metavar='val=expr', action='append',
                                 help='python code for manipulating data within page generation.')
        self.add_option_argument('--pdf', metavar='pdf', help='generate pdf.')
        self.add_option_argument('--pandas', metavar='pandas', action='append',
                                 help='pandas plots.')
        self.add_option_argument('--legend', metavar='legend', action='append',
                                 help='plot legend')
        self.add_option_argument('--grid', metavar='grid', action='append',
                                 help='grid for plotting.')
        self.add_option_argument('--subplot', metavar='subplot', action='append',
                                 help='define subplot.')
        self.add_option_argument('--figure', metavar='figure function', action='append',
                                 help='define Figure function.')
        self.add_option_argument('--axes', metavar='axes', action='append',
                                 help='define Axes function.')
        self.add_option_argument('--noshow', action='store_true', default=False,
                                 help='prevent showing plot on screen.')
        self.add_option_argument('--version', action='version',
                                 version='matplotlib plotting task version 0.1.0')


    def perform(self, targs):

        # pdf setting
        _pdf = None
        if targs.pdf:
            eval_vargs = self.teval(targs.pdf.vargs, autoimport=True)
            eval_kwargs = self.teval(targs.pdf.kwargs, autoimport=True)
            from matplotlib.backends.backend_pdf import PdfPages
            _pdf = PdfPages(*eval_vargs, **eval_kwargs)

        # pages setting
        if targs.pages:

            # syntax: <numpage> **kwarg
            eval_vargs = self.teval(targs.pages.vargs, autoimport=True)
            eval_kwargs = self.teval(targs.pages.kwargs, autoimport=True)

            if eval_vargs:
                self.env['num_pages'] = eval_vargs[-1]
            else:
                self.env['num_pages'] = 1

            for key, value in eval_kwargs.items():
                self.env[key] = value
        else:
            self.env['num_pages'] = 1

        # page iteration
        for idx in range(self.env['num_pages']):

            self.env['page_num'] = idx

            if targs.page_calc:
                for opt in targs.page_calc:

                    # syntax: <name>=value, ... 
                    eval_kwargs = self.teval(targs.pages_calc.kwargs, autoimport=True)
                    self.env.update(eval_kwargs)

            # figure setting
            if targs.f:

                eval_vargs = self.teval(targs.f.vargs, autoimport=True)
                eval_kwargs = self.teval(targs.f.kwargs, autoimport=True)

                self.env['figure'] = self.env['pyplot'].figure(*eval_vargs,
                                     **eval_kwargs)
            else:
                self.env['figure'] = self.env['pyplot'].figure()

            # plot axis
            if targs.subplot:

                for opt in targs.subplot:

                    # syntax: subplotname@funcargs
                    # future syntax: funcargs@subplotname
                    # Exploration over exploitation
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    if len(opt.context) == 1:

                        subpname = opt.context[0]

                        if 'projection' in eval_kwargs and eval_kwargs['projection'] == '3d':
                             from mpl_toolkits.mplot3d import Axes3D
                             self.env['Axes3D'] = Axes3D
                        if eval_vargs:
                            subplot = self.env['figure'].add_subplot(*eval_vargs, **eval_kwargs)
                        else:
                            subplot = self.env['figure'].add_subplot(111, **eval_kwargs)

                        try:
                            self.env[subpname] = subplot
                        except Exception as err:
                            self.parent.error_exit("syntax error from subplot name: %s"%subpname)
                    else:
                        self.parent.error_exit("Exactly one axis name is required at subplot option")

            # page names
            if 'page_names' in self.env:
                page_names = self.env['page_names']
                if callable(page_names):
                    self.env['page_name'] = self.teval(page_names)
                else:
                    self.env['page_name'] = page_names[self.env['page_num']]
            else:
                self.env['page_name'] = 'page%d'%self.env['page_num']

            # execute figure functions
            if targs.figure:

                for opt in targs.figure:

                    # syntax: funcname@funcargs
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    if len(opt.context) == 1:
                        getattr(self.env['figure'], opt.context[0])(*eval_vargs, **eval_kwargs)
                    else:
                        self.parent.error_exit("The synaxt error at figure option")

            if targs.pandas:

                for opt in targs.pandas:

                    # syntax: data@[plot@]funcargs
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    if len(opt.context) == 1:
                        data = self.teval(opt.context[0])
                        plot = getattr(data, "plot")
                    elif len(opt.context) == 2:
                        data = self.teval(opt.context[0])
                        plot = getattr(data, "plot")
                        plot = getattr(plot, opt.context[1])
                    else:
                        self.parent.error_exit("The synaxt error at pandas option")

                    subp = plot(*eval_vargs, **eval_kwargs)
                    if "ax" not in self.env:
                        self.env['ax'] = subp

            elif not targs.subplot:
                self.env['ax'] = self.env['figure'].add_subplot(111)

            # plotting
            plots = []
            if targs.plot:
                for opt in targs.plot:

                    # syntax: [axname@]funcname@funcargs

                    #evalopt = self.teval_(opt, True, False)
                    if len(opt.context) == 1:
                        axis = self.env["ax"]
                        funcname = opt.context[0]
                    elif len(opt.context) == 2:
                        axis = self.env[opt.context[0]]
                        funcname = opt.context[1]
                    else:
                        self.parent.error_exit("The synaxt error at plot option")

                    if hasattr(axis, funcname):

                        try:
                            eval_vargs = self.teval(opt.vargs, autoimport=True)
                            eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                            plot_handle = getattr(axis, funcname)(*eval_vargs, **eval_kwargs)

                            try:
                                for p in plot_handle:
                                    plots.append(p)
                            except TypeError:
                                plots.append(plot_handle)
                        except Exception as err:
                            print("WARNING: %s"%str(err))
                    else:
                        # TODO: handling this case
                        pass

                    if funcname == 'pie':
                        axis.axis('equal')

            if 'plots' in self.env:
                self.env['plots'].extend(plots)
            else:
                self.env['plots'] = plots

            # title setting
            if targs.title:
                # syntax: [axname@]funcargs
                opt = targs.title

                eval_vargs = self.teval(opt.vargs, autoimport=True)
                eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                if len(opt.context) == 0:
                    axes = [self.env["ax"]]
                elif len(opt.context) == 1:
                    axes = [self.env[opt.context[0]]]
                else:
                    self.parent.error_exit("The synaxt error at title option")

                for ax in axes:
                    ax.set_title(*eval_vargs, **eval_kwargs)

            # x-axis setting
            if targs.xaxis:

                for opt in targs.xaxis:

                    # syntax: [axname@]funcname@funcargs
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    if len(opt.context) == 1:
                        axis = self.env["ax"]
                        funcname = "set_x"+opt.context[0]
                    elif len(opt.context) == 2:
                        axis = self.env[opt.context[0]]
                        funcname = "set_x"+opt.context[1]
                    else:
                        self.parent.error_exit("The synaxt error at xaxis option")

                    if hasattr(axis, funcname):
                        getattr(axis, funcname)(*eval_vargs, **eval_kwargs)
                    else:
                        # TODO: handling this case
                        pass

            # y-axis setting
            if targs.yaxis:
                for opt in targs.yaxis:

                    # syntax: [axname@]funcname@funcargs
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    if len(opt.context) == 1:
                        axis = self.env["ax"]
                        funcname = "set_y"+opt.context[0]
                    elif len(opt.context) == 2:
                        axis = self.env[opt.context[0]]
                        funcname = "set_y"+opt.context[1]
                    else:
                        self.parent.error_exit("The synaxt error at yaxis option")

                    if hasattr(axis, funcname):
                        getattr(axis, funcname)(*eval_vargs, **eval_kwargs)
                    else:
                        # TODO: handling this case
                        pass

            # z-axis setting
            if targs.zaxis:
                for opt in targs.zaxis:

                    # syntax: [axname@]funcname@funcargs
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    if len(opt.context) == 1:
                        axis = self.env["ax"]
                        funcname = "set_z"+opt.context[0]
                    elif len(opt.context) == 2:
                        axis = self.env[opt.context[0]]
                        funcname = "set_z"+opt.context[1]
                    else:
                        self.parent.error_exit("The synaxt error at zaxis option")

                    if hasattr(axis, funcname):
                        getattr(axis, funcname)(*eval_vargs, **eval_kwargs)
                    else:
                        # TODO: handling this case
                        pass

            # grid setting
            if targs.g:
                for key, value in self.env.items():
                    if isinstance(value, self.env['mpl'].axes.Axes):
                        value.grid()

            if targs.grid:

                for opt in targs.grid:

                    # syntax: [axname@]funcargs
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    if len(opt.context) == 0:
                        axis = self.env["ax"]
                    elif len(opt.context) == 1:
                        axis = self.env[opt.context[0]]
                    else:
                        self.parent.error_exit("The synaxt error at grid option")

                    axis.grid(*eval_vargs, **eval_kwargs)

            # legend setting
            if targs.l:
                for key, value in self.env.items():
                    if isinstance(value, self.env['mpl'].axes.Axes):
                        value.legend()

            if targs.legend:
                for opt in targs.legend:

                    # syntax: [axname@]funcargs
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    if len(opt.context) == 0:
                        axis = self.env["ax"]
                    elif len(opt.context) == 1:
                        axis = self.env[opt.context[0]]
                    else:
                        self.parent.error_exit("The synaxt error at legend option")

                    axis.legend(*eval_vargs, **eval_kwargs)

            # execute axes functions
            if targs.axes:
                for opt in targs.axes:
                    # syntax: [axname@]funcname@funcargs
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    if len(opt.context) == 1:
                        axis = self.env["ax"]
                        funcname = opt.context[0]
                    elif len(opt.context) == 2:
                        axis = self.env[opt.context[0]]
                        funcname = opt.context[1]
                    else:
                        self.parent.error_exit("The synaxt error at axes option")

                    getattr(axis, funcname)(*eval_vargs, **eval_kwargs)

            elif not self.env['plots']:
                if targs.figure or targs.pandas:
                    pass
                elif self.env["D"]:
                    for d in self.env["D"]:
                        self.env["ax"].plot(d)
                else:
                    self.parent.error_exit("There is no data to plot.")

            # saving an image file
            if targs.save:

                for opt in targs.save:

                    #vargs, kwargs = self.teval_args(save_arg)
                    eval_vargs = self.teval(opt.vargs, autoimport=True)
                    eval_kwargs = self.teval(opt.kwargs, autoimport=True)

                    name = eval_vargs.pop(0)

                    if self.env['num_pages'] > 1:
                        if os.path.exists(name):
                            if not os.path.isdir(name):
                                os.remove(name)
                                os.makedirs(name)
                        else:
                            os.makedirs(name)

                        root, ext = os.path.splitext(name)
                        name = os.path.join(root, str(self.env['page_num'])+ext)

                    self.env["figure"].savefig(name, *eval_vargs, **eval_kwargs)

            # displyaing an image on screen
            if not targs.noshow:
                self.env['pyplot'].show()

            if _pdf is not None:
                _pdf.savefig()

            self.env["figure"].clear()
            self.env["pyplot"].close(self.env["figure"])
            del self.env['figure']

        if _pdf is not None:
            _pdf.close()

        return 0


