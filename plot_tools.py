"""Quick plotting tools go here

NOte: older code from Rupert used pylab, which is now discouraged. KD changed to pyplot on 7-10-19 but has
not tested all older aspects of the code to ensure proper switch from pylab (though should be the same)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from mm_params import tp, sp, iop, ap
from mm_utils import dprint
import colormaps as cmaps

plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)
plt.register_cmap(name='inferno', cmap=cmaps.plasma)
plt.register_cmap(name='magma', cmap=cmaps.plasma)


# MEDIUM_SIZE = 17
# plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes

from matplotlib import rcParams
rcParams['font.family'] = 'STIXGeneral'  # 'Times New Roman'
# rcParams['mathtext.fontset'] = 'custom'
# rcParams['mathtext.fontset'] = 'stix'
# rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


def quick2D(image, title=None, logAmp=False, vlim=(None,None), colormap=None):
    """
    Looks at a 2D array, has bunch of handles for plot.imshow

    :param image: 2D array to plot (data)
    :param title: string--must be set or will error!
    :param logAmp: flag to set logscale plotting in amplitude maps.
    :param vlim: tuple of limits on the colorbar axis, otherwise default matplotlib (pass in logscale limits if LogAmp=True)
    :param colormap: specify colormap as string
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Title
    while title is None:
        print("Plots without titles: Don't Do It!")
        title = input("Please Enter Title: ")

    # Setting Colorbar limits
    vmin = vlim[0]
    vmax = vlim[1]

    # Setting Logscale
    if logAmp:
        if np.min(image) <= 0:
            cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
                            norm=SymLogNorm(linthresh=1e-5),
                            cmap="YlGnBu_r")
            clabel = "Log Normalized Intensity"
        else:
            cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
                            norm=LogNorm(), cmap="YlGnBu_r")
            clabel = "Log Normalized Intensity"
    else:
        cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
        clabel = "Normalized Intensity"

    # Plotting
    plt.title(title, fontweight='bold', fontsize=16)
    cb = plt.colorbar(cax)
    cb.set_label(clabel)
    plt.show()


def view_datacube(datacube, title=None, show=True, logAmp=False, use_axis=True, vlim =(None,None), subplt_cols=3):
    """
    view plot of each wavelength bin intensity at a single timestep

    :param datacube: 3D spectral cube at single timestep
    :param title: string, must be set or will error!
    :param show: flag possibly useful for plotting loops of things?
    :param logAmp: turn logscale plotting on or off
    :param use_axis: turn on/off using axis ticks, colorbar, etc
    :param vlim: tuple of colorbar axis limits (min,max)
    :param xylims: range of x-y axis to plot (inner portion of grid)
    :param subplt_cols: number of subplots per row
    :return:
    """
    plt.close('all')

    w_string = np.array(np.linspace(ap.wvl_range[0]*1e9, ap.wvl_range[1]*1e9, ap.w_bins, dtype=int), dtype=str)

    # Number of subplots size
    fig = plt.figure()
    n_colors = len(datacube)
    n_rows = int(np.ceil(n_colors / float(subplt_cols))+1)
    plt.axis('off')
    gs = gridspec.GridSpec(n_rows, subplt_cols, wspace=0.08, top=0.9, bottom=0.2)

    # Title
    if title is None:
        raise NameError("Plots without titles: Don't Do It!")
        title = input("Please Enter Title: ")
        pass
    fig.suptitle(title, fontweight='bold', fontsize=16)

    # Checking Colorbar axis limits
    vmin = vlim[0]
    vmax = vlim[1]
    # peak = np.max(datacube)
    # trough = np.min(datacube)
    # dprint{f"peak={peak}, trough={trough}")
    # if vmin != None:
    #     if peak >= vmin:
    #         vmax = peak
    # else:
    #     vmin = trough

    for w in range(n_colors):
        # ax = fig.add_subplot(n_rows, subplt_cols, w + 1)
        ax = fig.add_subplot(gs[w])
        if logAmp:
            if vmin is not None and vmin <= 0:
                ax.set_title(r'$\lambda$ = '+f"{w_string[w]} nm")
                im = ax.imshow(datacube[w], interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
                               norm=SymLogNorm(linthresh=1e-5),
                               cmap="YlGnBu_r")
                clabel = "Log Normalized Intensity"
            else:
                ax.set_title(r'$\lambda$ = '+f"{w_string[w]} nm")
                im = ax.imshow(datacube[w], interpolation='none', origin='lower', vmin=vmin, vmax=vmax, norm=LogNorm(),
                               cmap="YlGnBu_r")
                clabel = "Log Normalized Intensity"
        else:
            ax.set_title(r'$\lambda$ = '+f"{w_string[w]} nm")
            im = ax.imshow(datacube[w], interpolation='none', origin='lower', vmin=vmin, vmax=vmax, cmap="YlGnBu_r")
            clabel = "Normalized Intensity"

        if use_axis == 'anno':
            annotate_axis(im, ax, datacube.shape[1])
        if use_axis is None:
            plt.axis('off')

    if use_axis:
        gs.tight_layout(fig, pad=0.08, rect=(0, 0, 1, 0.9))  # rect = (left, bottom, right, top)
        # fig.tight_layout(pad=50)
        cbar_ax = fig.add_axes([0.55, 0.05, 0.2, 0.05])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')  #
        cb.set_label(clabel)

    if show is True:
        plt.show(block=True)


# def fmt(x, pos):
#     a, b = '{:.0e}'.format(x).split('e')
#     b = int(b)
#     return r'${} e^{{{}}}$'.format(a, b)
#
#
# def grid(datacube, nrows=2, logAmp=False, axis=None, width=None, titles=None, ctitles=None, annos=None,
#          scale=1, vmins=None, vmaxs=None, show=True):
#     import matplotlib
#     # dprint(matplotlib.is_interactive())
#     matplotlib.interactive(1)
#     MEDIUM_SIZE = 14
#     plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
#     '''axis = anno/None/True'''
#     if not width:
#         width = len(datacube) // nrows
#     if vmins is None:
#         vmins = len(datacube) * [None]
#         vmaxs = len(datacube) * [None]
#
#     # dprint((nrows, width))
#     fig, axes = plt.subplots(nrows=nrows, ncols=width, figsize=(14, 8))
#     axes = axes.reshape(nrows, width)
#     labels = ['e', 'f', 'g', 'h']
#     m = 0
#
#     # for left right swap x and y
#     for x in range(width):
#         for y in range(nrows):
#
#             if logAmp:
#                 if np.min(datacube[m]) <= 0:
#                     # datacube[m] = np.abs(datacube[m]) + 1e-20
#                     im = axes[y, x].imshow(datacube[m], interpolation='none', origin='lower', vmin=vmins[m],
#                                            vmax=vmaxs[m], norm=SymLogNorm(linthresh=1e-7), cmap="YlGnBu_r")
#                 else:
#                     im = axes[y, x].imshow(datacube[m], interpolation='none', origin='lower', vmin=vmins[m],
#                                            vmax=vmaxs[m], norm=LogNorm(), cmap="YlGnBu_r")
#             else:
#                 im = axes[y, x].imshow(datacube[m], interpolation='none', origin='lower',
#                                        vmin=vmins[m], vmax=vmaxs[m], cmap='viridis')  # "YlGnBu_r"
#             if titles is not None:
#                 axes[y, x].set_title(str(titles[m]))
#             props = dict(boxstyle='square', facecolor='k', alpha=0.5)
#             if annos is not None:
#                 axes[y, x].text(0.05, 0.075, annos[m], transform=axes[y, x].transAxes, fontweight='bold',
#                                 color='w', fontsize=22, bbox=props)
#             if axis == 'anno':
#                 annotate_axis(im, axes[y, x], datacube.shape[1])
#             if axis is None:
#                 axes[y, x].axis('off')
#
#             # if y ==1:
#             #     circle1 = plt.Circle((124, 56), radius=6, color='g', fill=False, linewidth=2)
#             #     circle2 = plt.Circle((104,119), radius=6, color='g', fill=False, linewidth=2)
#             #     axes[y, x].add_artist(circle1)
#             #     axes[y, x].add_artist(circle2)
#             # axes[y, x].arrow(105, 43, -10, 0, head_width=5, head_length=3, fc='r', ec='r')
#             # axes[y, x].arrow(122,28, -10, 0, head_width=5, head_length=3, fc='r', ec='r')
#             # if m == 2 or m==6:
#             #     axes[y, x].arrow(114, 33, -10, 0, head_width=5, head_length=3, fc='r', ec='r')
#             # else:
#             #     axes[y, x].arrow(103.5, 43, -10, 0, head_width=5, head_length=3, fc='r', ec='r')
#
#             # axes[y, x].text(0.05, 0.85, labels[m], transform=axes[y,x].transAxes, fontweight='bold', color='w', fontsize=22, family='serif',bbox=props)
#             m += 1
#
#         if ctitles and nrows == 2 and width == 2:
#             cax = fig.add_axes([0.9, 0.01, 0.03, 0.89])
#             # cb = fig.colorbar(im, cax=cax, orientation='vertical',format=ticker.FuncFormatter(fmt))
#             cb = fig.colorbar(im, cax=cax, orientation='vertical')
#             # cb.ax.set_title(ctitles[y], fontsize=16)
#             cb.ax.set_title(ctitles, fontsize=20)
#
#         if ctitles and width == 1:
#             cax = fig.add_axes([0.8, 0.02, 0.05, 0.89])
#             # cb = fig.colorbar(im, cax=cax, orientation='vertical',format=ticker.FuncFormatter(fmt))
#             cb = fig.colorbar(im, cax=cax, orientation='vertical')
#             # cb.ax.set_title(ctitles[y], fontsize=16)
#             cb.ax.set_title(ctitles, fontsize=20)
#
#     if show:
#         plt.subplots_adjust(left=0.01, right=0.86, top=0.9, bottom=0.01, wspace=0.1, hspace=0.1)
#         plt.show(block=True)
#
#
# def indep_images(datacube, logAmp=False, axis=None, width=None, titles=None, annos=None, scale=1, vmins=None,
#                  vmaxs=None):
#     MEDIUM_SIZE = 14
#     plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
#     '''Like compare_images but independent scaling colorbars'''
#     '''axis = anno/None/True'''
#     if not width:
#         width = len(datacube)
#     if vmins is None:
#         vmins = len(datacube) * [None]
#         vmaxs = len(datacube) * [None]
#
#     fig, axes = plt.subplots(nrows=1, ncols=width, figsize=(14, 10))
#
#     labels = ['a', 'b', 'c', 'd', 'e', 'f']
#     for m, ax in enumerate(axes):
#         if logAmp:
#             if np.min(datacube[m]) <= 0:
#                 # datacube[m] = np.abs(datacube[m]) + 1e-20
#                 im = ax.imshow(datacube[m], interpolation='none', origin='lower', vmin=vmins[m], vmax=vmaxs[m],
#                                norm=SymLogNorm(linthresh=1e-7), cmap="YlGnBu_r")
#             else:
#                 im = ax.imshow(datacube[m], interpolation='none', origin='lower', vmin=vmins[m],
#                                vmax=vmaxs[m], norm=LogNorm(), cmap="YlGnBu_r")
#
#         else:
#             im = ax.imshow(datacube[m], interpolation='none', origin='lower', vmin=vmins[m],
#                            vmax=vmaxs[m], cmap="YlGnBu_r")
#         props = dict(boxstyle='square', facecolor='k', alpha=0.5)
#         if annos:
#             ax.text(0.05, 0.05, annos[m], transform=ax.transAxes, fontweight='bold', color='w', fontsize=22, bbox=props)
#         if axis == 'anno':
#             annotate_axis(im, ax, datacube.shape[1])
#         if axis is None:
#             ax.axis('off')
#         if titles:
#             # cax = fig.add_axes([0.27+ 0.335*m, 0.01, 0.01, 0.89])
#             if width == 2:
#                 cax = fig.add_axes([0.4 + 0.5 * m, 0.01, 0.02, 0.89])
#             if width == 3:
#                 cax = fig.add_axes([0.27 + 0.33 * m, 0.04, 0.015, 0.86])
#             # cb = fig.colorbar(im, cax=cax, orientation='vertical',format=ticker.FuncFormatter(fmt))
#             cb = fig.colorbar(im, cax=cax, orientation='vertical')
#
#             cb.ax.set_title(titles[m], fontsize=16)
#         # circle1 = plt.Circle((86, 43), radius=4, color='w', fill=False, linewidth=1)
#         # circle2 = plt.Circle((103,28), radius=4, color='w', fill=False, linewidth=1)
#         # ax.add_artist(circle1)
#         # ax.add_artist(circle2)
#         ax.arrow(105, 43, -10, 0, head_width=5, head_length=3, fc='r', ec='r')
#         ax.arrow(122, 28, -10, 0, head_width=5, head_length=3, fc='r', ec='r')
#
#         # ax.text(0.05, 0.9, labels[m], transform=ax.transAxes, fontweight='bold', color='w', fontsize=22, family='serif',bbox=props)
#     axes[0].text(0.84, 0.9, '0.2"', transform=axes[0].transAxes, fontweight='bold', color='w', ha='center', fontsize=14,
#                  family='serif')
#     axes[0].plot([0.78, 0.9], [0.87, 0.87], transform=axes[0].transAxes, color='w', linestyle='-', linewidth=3)
#
#     # if cbar:
#     #     plt.ticklabel_format(useOffset=False)
#     #     if width != 2:
#     #         cax = fig.add_axes([0.94, 0.01, 0.01, 0.87])
#     #     elif width ==2:
#     #         cax = fig.add_axes([0.84, 0.01, 0.02, 0.89])
#     #     cb = fig.colorbar(im, cax=cax, orientation='vertical')
#     #     if titles:
#     #         cb.ax.set_title(r'  $I / I^{*}$', fontsize=16)
#
#     plt.subplots_adjust(left=0.01, right=0.93, top=0.9, bottom=0.01, wspace=0.33)
#     plt.show()
#
#
# def compare_images(datacube, logAmp=False, axis=None, width=None, title=None, annos=None, scale=1, max_scale=0.05,
#                    vmin=None, vmax=None):
#     MEDIUM_SIZE = 14
#     plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
#     '''Like view datacube by colorbar on the right and apply annotations'''
#     '''axis = anno/None/True'''
#     if not width:
#         width = len(datacube)
#     if title is None:
#         title = r'  $I / I^{*}$'
#     # fig =plt.figure(figsize=(14,7))
#
#     if width == 4 or width != 2:
#         fig, axes = plt.subplots(nrows=2, ncols=width, figsize=(14, 3.4))
#     elif width == 2:
#         fig, axes = plt.subplots(nrows=2, ncols=width, figsize=(7, 3.1))
#     else:
#         fig, axes = plt.subplots(nrows=1, ncols=width, figsize=(14, 8))
#         axes = axes.reshape(1, width)
#     # maps = len(datacube)
#
#     # norm = np.sum(datacube[0])
#     # datacube = datacube/norm
#
#     peaks, troughs = [], []
#     dprint(f"datacube shape={datacube.shape}, axis shape={axes.shape}, width={width}")
#     for image in datacube:
#         peaks.append(np.max(image))
#         troughs.append(np.min(image))
#
#     if vmin is None:
#         vmin = np.min(troughs)
#         # if vmin<=0:
#         #     troughs = np.array(troughs)
#         #     print(troughs)
#         #     vmin = min(troughs[troughs>=0])+1e-20
#         vmin *= scale
#
#     if vmax is None:
#         vmax = np.max(peaks)
#         dprint((vmin, vmax))
#         if max_scale:
#             vmax *= max_scale
#     # if vmax <= 0: vmax = np.abs(vmax) + 1e-20
#
#     # labels = ['a','b','c','d','e']
#     # labels = ['a i', 'ii', 'iii', 'iv', 'v', 'vi']
#     labels = list(range(width))
#     for m, ax in enumerate(axes):
#         # ax = fig.add_subplot(1,width,m+1)
#         # axes.append(ax)
#         if logAmp:
#             if np.min(datacube[m]) <= 0.:
#                 im = ax[m].imshow(datacube[m], interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
#                                   norm=SymLogNorm(linthresh=1e-7), cmap="YlGnBu_r")
#                 # datacube[m] = np.abs(datacube[m]) + 1e-20
#                 dprint('corrected', np.min(datacube[m]), np.max(datacube[m]), vmin, vmax)
#             else:
#                 im = ax.imshow(datacube[m], interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
#                                norm=LogNorm(), cmap="YlGnBu_r")
#         else:
#             im = ax.imshow(datacube[m], interpolation='none', origin='lower', vmin=vmin, vmax=vmax, cmap="jet")
#         if annos:
#             ax.text(0.05, 0.05, annos[m], transform=ax.transAxes, fontweight='bold', color='w', fontsize=22)
#         if axis == 'anno':
#             annotate_axis(im, ax, datacube.shape[1])
#         if axis is None:
#             ax.axis('off')
#         # ax.plot(image.shape[0] / 2, image.shape[1] / 2, marker='*', color='r')
#         # import matplotlib.patches as patches#40,100,20,60
#         # rect = patches.Rectangle((68, 28), 40, 60, linewidth=1, edgecolor='r', facecolor='none')
#         # ax.add_patch(rect)
#         # if m != 2:
#         #     ax.arrow(103.5, 43, -10, 0, head_width=5, head_length=3, fc='r', ec='r')
#         # else:
#         #     ax.arrow(114, 33, -10, 0, head_width=5, head_length=3, fc='r', ec='r')
#         # ax.grid(color='w', linestyle='--')
#         # circle1 = plt.Circle((2, 34), radius=4, color='w', fill=False, linewidth=2)
#         # ax.add_artist(circle1)
#         ax.text(0.04, 0.9, labels[m], transform=ax.transAxes, fontweight='bold', color='w', fontsize=22, family='serif')
#
#     if width == 3:
#         cax = fig.add_axes([0.9, 0.01, 0.015, 0.87])
#     elif width == 2:
#         cax = fig.add_axes([0.84, 0.01, 0.02, 0.89])
#     else:
#         # cax = fig.add_axes([0.94, 0.01, 0.01, 0.87])
#         cax = fig.add_axes([0.94, 0.04, 0.01, 0.86])
#     cb = fig.colorbar(im, cax=cax, orientation='vertical', norm=LogNorm(), format=ticker.FuncFormatter(fmt))
#     # cb = fig.colorbar(im, cax=cax, orientation='vertical', format=ticker.FuncFormatter(fmt))
#     cb.ax.set_title(title, fontsize=16)  #
#     cbar_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5, endpoint=True)
#     cb.set_ticks(cbar_ticks)
#     # cbar_ticks = [1e-6, 1e-5, 1e-4]#np.linspace(1e-7, 1e-4, num=4, endpoint=True)
#     # cb.set_ticks(cbar_ticks)
#
#     if width != 2:
#         plt.subplots_adjust(left=0.01, right=0.92, top=0.9, bottom=0.01, wspace=0.12)
#     elif width == 2:
#         plt.subplots_adjust(left=0.01, right=0.82, top=0.9, bottom=0.01, wspace=0.05)
#     plt.show()
#
#
# def get_intensity(wf_array, sp, logAmp=True, show=False, save=True, phase=False):
#     if show is True:
#         wfo = wf_array[0, 0]
#         after_dm = proper.prop_get_amplitude(wfo)
#         phase_afterdm = proper.prop_get_phase(wfo)
#
#         fig = plt.figure(figsize=(14, 10))
#         ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
#         ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
#         ax3 = plt.subplot2grid((3, 2), (2, 0))
#         ax4 = plt.subplot2grid((3, 2), (2, 1))
#         if logAmp:
#             ax1.imshow(after_dm, origin='lower', cmap="YlGnBu_r", norm=LogNorm())
#         else:
#             ax1.imshow(after_dm, origin='lower', cmap="YlGnBu_r")
#         ax2.imshow(phase_afterdm, origin='lower', cmap="YlGnBu_r")  # , vmin=-0.5, vmax=0.5)
#
#         ax3.plot(after_dm[int(tp.grid_size / 2)])
#         ax3.plot(np.sum(np.eye(tp.grid_size) * after_dm, axis=1))
#
#         # plt.plot(np.sum(after_dm,axis=1)/after_dm[128,128])
#
#         ax4.plot(phase_afterdm[int(tp.grid_size / 2)])
#         # ax4.plot(np.sum(np.eye(tp.grid_size)*phase_afterdm,axis=1))
#         plt.xlim([0, proper.prop_get_gridsize(wfo)])
#         fig.set_tight_layout(True)
#
#         plt.show()
#
#     if save:
#         ws = sp.get_ints['w']
#         cs = sp.get_ints['c']
#
#         int_maps = np.empty((0, tp.grid_size, tp.grid_size))
#         for iw in ws:
#             for iwf in cs:
#                 # int_maps.append(proper.prop_shift_center(np.abs(wf_array[iw, iwf].wfarr) ** 2))
#                 if phase:
#                     int_map = proper.prop_get_phase(wf_array[iw, iwf])
#
#                 # int_map = proper.prop_shift_center(np.abs(wf_array[iw, iwf].wfarr) ** 2)
#                 else:
#                     int_map = proper.prop_shift_center(np.abs(wf_array[iw, iwf].wfarr) ** 2)
#                 int_maps = np.vstack((int_maps, [int_map]))
#                 # quicklook_im(int_map)#, logAmp=True)
#
#         import pickle, os
#         if os.path.exists(iop.int_maps):
#             # "with" statements are very handy for opening files.
#             with open(iop.int_maps, 'rb') as rfp:
#                 # dprint(np.array(pickle.load(rfp)).shape)
#                 int_maps = np.vstack((int_maps, pickle.load(rfp)))
#
#         with open(iop.int_maps, 'wb') as wfp:
#             pickle.dump(int_maps, wfp, protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
# def initialize_GUI():
#     # plt.ion()
#     sp.show_wframe = 'continuous'
#     sp.fig = plt.figure()
#     # ax = sp.fig.add_subplot(111)
#     # ax.plot(range(5))
#
#
#
# def quicklook_im(image, logAmp=False, show=True, vmin=None, vmax=None, axis=False, anno=None, title=None, pupil=False,
#                  colormap="YlGnBu_r", mark_star=False, label=None, block=False):
#     """
#     Looks at a 2D array, has bunch of handles for plot.imshow
#
#     :param image: 2D array to plot (data)
#     :param logAmp:
#     :param show:
#     :param vmin: colorbar axis limits (min)
#     :param vmax: colorbar axis limits (max)
#     :param axis:
#     :param anno:
#     :param title:
#     :param pupil: flag to change the colorscheme whether the image is a focal plane image or pupil plane image
#     :param colormap:
#     :param mark_star: flag to mark the location of central star (useful if coronagraph)
#     :param label:
#     :return:
#     """
#     fig = plt.figure()
#     if pupil:
#         import medis.Analysis.phot
#         image = image * Analysis.phot.aperture(tp.grid_size / 2, tp.grid_size / 2, tp.grid_size / 2)
#
#     if title is None:
#         title = r'  $I / I^{*}$'
#
#     ax = fig.add_subplot(111)
#     if logAmp:
#         if np.min(image) <= 0:
#
#             cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
#                             norm=SymLogNorm(linthresh=1e-5),
#                             cmap="YlGnBu_r")
#         else:
#             cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vmin, vmax=vmax,
#                             norm=LogNorm(), cmap="YlGnBu_r")
#
#     else:
#         cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vmin, vmax=vmax, cmap=colormap)
#     if axis:
#         annotate_axis(cax, ax, image.shape[0])
#
#     cb = plt.colorbar(cax)
#
#     if axis is None:
#         ax.axis('off')
#
#     # if anno:
#     #     props = dict(boxstyle='square', facecolor='k', alpha=0.3)
#     #     ax.text(0.05, 0.05, anno, transform=ax.transAxes, fontweight='bold', color='w', fontsize=22, bbox=props)
#     # if label:
#     #     props = dict(boxstyle='square', facecolor='k', alpha=0.3)
#     #     ax.text(0.05, 0.9, label, transform=ax.transAxes, fontweight='bold', color='w', fontsize=22, family='serif',
#     #             bbox=props)
#     # # ax.text(0.84, 0.9, '0.5"', transform=ax.transAxes, fontweight='bold', color='w', ha='center', fontsize=14, family='serif')
#     # # ax.plot([0.78, 0.9], [0.87, 0.87],transform=ax.transAxes, color='w', linestyle='-', linewidth=3)
#     # if mark_star:
#     #     ax.plot(image.shape[0] / 2, image.shape[1] / 2, marker='*', color='r')
#
#     # # For plotting on the leftmost screen
#     # figManager = plt.get_current_fig_manager()
#     # # if px=0, plot will display on 1st screen
#     # figManager.window.move(-1920, 0)
#     # figManager.window.setFocus()
#
#     if show:
#         #plt.tight_layout()
#         plt.show()
#
#
# def annotate_axis(im, ax, width):
#     rad = tp.platescale / 1000 * width / 2
#     ticks = np.linspace(-rad, rad, 5)
#     ticklabels = ["{:0.3f}".format(i) for i in ticks]
#     ax.set_xticks(np.linspace(-0.5, width - 0.5, 5))  # -0.5
#     ax.set_yticks(np.linspace(-0.5, width - 0.5, 5))  # -0.5
#     ax.set_xticklabels(ticklabels)
#     ax.set_yticklabels(ticklabels)
#     im.axes.tick_params(color='white', direction='in', which='both', right=True, top=True, width=2,
#                         length=10)  # , labelcolor=fg_color)
#     im.axes.tick_params(which='minor', length=5)
#     ax.xaxis.set_minor_locator(ticker.FixedLocator(np.linspace(-0.5, width - 0.5, 33)))
#     ax.yaxis.set_minor_locator(ticker.FixedLocator(np.linspace(-0.5, width - 0.5, 33)))
#
#     ax.set_xlabel('RA (")')
#     ax.set_ylabel('Dec (")')
#
#
# def loop_frames(frames, axis=False, circles=None, logAmp=False, vmin=None, vmax=None, show=True):
#     fig, ax = plt.subplots()
#     ax.set(title='Click to update the data')
#     dprint((vmin, vmax))
#     if logAmp:
#         if np.min(frames) < 0:
#             im = ax.imshow(frames[0], norm=SymLogNorm(linthresh=1e-3), origin='lower', vmin=vmin, vmax=vmax)
#         else:
#             im = ax.imshow(frames[0], norm=LogNorm(), origin='lower', vmin=vmin, vmax=vmax)
#     else:
#         im = ax.imshow(frames[0], origin='lower', vmin=vmin, vmax=vmax)
#     if axis:
#         annotate_axis(im, ax, frames.shape[1])
#     cbar = plt.colorbar(im)
#     # if logAmp:
#     #     cbar_ticks = np.logspace(np.min(frames), np.max(frames), num=7, endpoint=True)
#     if not logAmp:
#         cbar_ticks = np.linspace(np.min(frames), np.max(frames), num=13, endpoint=True)
#         cbar.set_ticks(cbar_ticks)
#
#     # serious python kung fu to get the click update working. probably there is an easier way
#     class frame():
#         def __init__(self):
#             self.f = 1
#
#         def update(self, event):
#             print(self.f, len(frames))
#             if self.f == len(frames):
#                 print('Cannot cycle any more')
#             else:
#                 im.set_data(frames[self.f])
#                 # cbar.set_clim(np.min(frames[self.f]), np.max(frames[self.f]))
#                 # dprint(im.vmax)
#                 # if logAmp:
#                 #     cbar_ticks = np.logspace(np.min(frames[self.f]), np.max(frames[self.f]), num=7, endpoint=True)
#                 # if not logAmp:
#                 #     cbar_ticks = np.linspace(np.min(frames[self.f]), np.max(frames[self.f]), num=7, endpoint=True)
#                 #     cbar.set_ticks(cbar_ticks)
#                 self.f += 1
#                 fig.canvas.draw()
#
#     frame = frame()
#
#     fig.canvas.mpl_connect('button_press_event', frame.update)
#     if show:
#         plt.show()
#
#
# def quicklook_wf(wfo, logAmp=True, show=True):
#     after_dm = proper.prop_get_amplitude(wfo)
#     phase_afterdm = proper.prop_get_phase(wfo)
#
#     fig = plt.figure(figsize=(14, 10))
#     ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
#     ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
#     ax3 = plt.subplot2grid((3, 2), (2, 0))
#     ax4 = plt.subplot2grid((3, 2), (2, 1))
#     if logAmp:
#         ax1.imshow(after_dm, origin='lower', cmap="YlGnBu_r", norm=LogNorm())
#     else:
#         ax1.imshow(after_dm, origin='lower', cmap="YlGnBu_r")
#     ax2.imshow(phase_afterdm, origin='lower', cmap="YlGnBu_r")  # , vmin=-0.5, vmax=0.5)
#
#     ax3.plot(after_dm[int(tp.grid_size / 2)])
#     ax3.plot(np.sum(np.eye(tp.grid_size) * after_dm, axis=1))
#
#     # plt.plot(np.sum(after_dm,axis=1)/after_dm[128,128])
#
#     ax4.plot(phase_afterdm[int(tp.grid_size / 2)])
#     # ax4.plot(np.sum(np.eye(tp.grid_size)*phase_afterdm,axis=1))
#     plt.xlim([0, proper.prop_get_gridsize(wfo)])
#     fig.set_tight_layout(True)
#
#     if show:
#         plt.show()
#     # ans = raw_input('here')
#
#
# def quicklook_IQ(wfo, logAmp=False, show=True):
#     I = np.real(wfo.wfarr)
#     Q = np.imag(wfo.wfarr)
#
#     I = proper.prop_shift_center(I)
#     Q = proper.prop_shift_center(Q)
#
#     fig = plt.figure(figsize=(14, 10))
#     ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
#     ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
#     ax3 = plt.subplot2grid((3, 2), (2, 0))
#     ax4 = plt.subplot2grid((3, 2), (2, 1))
#     if logAmp:
#         ax1.imshow(I, origin='lower', cmap="YlGnBu_r", norm=LogNorm())
#     else:
#         ax1.imshow(I, origin='lower', cmap="YlGnBu_r")
#     ax2.imshow(Q, origin='lower', cmap="YlGnBu_r")  # , vmin=-0.5, vmax=0.5)
#
#     ax3.plot(I[int(tp.grid_size / 2)])
#     ax3.plot(np.sum(np.eye(tp.grid_size) * I, axis=1))
#
#     # plt.plot(np.sum(after_dm,axis=1)/after_dm[128,128])
#
#     ax4.plot(Q[int(tp.grid_size / 2)])
#     # ax4.plot(np.sum(np.eye(tp.grid_size)*phase_afterdm,axis=1))
#     plt.xlim([0, proper.prop_get_gridsize(wfo)])
#     fig.set_tight_layout(True)
#     if show == True:
#         plt.show()
#         # ans = raw_input('here')
#
#
# def add_subplot_axes(ax, rect, axisbg='w'):
#     fig = plt.gcf()
#     box = ax.get_position()
#     width = box.width
#     height = box.height
#     inax_position = ax.transAxes.transform(rect[0:2])
#     transFigure = fig.transFigure.inverted()
#     infig_position = transFigure.transform(inax_position)
#     x = infig_position[0]
#     y = infig_position[1]
#     width *= rect[2]
#     height *= rect[3]  # <= Typo was here
#     subax = fig.add_axes([x, y, width, height], axisbg=axisbg)
#     x_labelsize = subax.get_xticklabels()[0].get_size()
#     y_labelsize = subax.get_yticklabels()[0].get_size()
#     x_labelsize *= rect[2] ** 0.5
#     y_labelsize *= rect[3] ** 0.5
#     subax.xaxis.set_tick_params(labelsize=x_labelsize)
#     subax.yaxis.set_tick_params(labelsize=y_labelsize)
#     return subax
