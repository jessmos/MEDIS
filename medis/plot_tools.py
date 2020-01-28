"""Quick plotting tools go here

Note: older code from Rupert used pylab, which is now discouraged. KD changed to pyplot on 7-10-19 but has
not tested all older aspects of the code to ensure proper switch from pylab (though should be the same)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from medis.params import tp, sp, iop, ap, cdip
import medis.utils as mu
import medis.optics as opx
import medis.colormaps as cmaps

plt.register_cmap(name='viridis', cmap=cmaps.viridis)
plt.register_cmap(name='plasma', cmap=cmaps.plasma)
plt.register_cmap(name='inferno', cmap=cmaps.plasma)
plt.register_cmap(name='magma', cmap=cmaps.plasma)

# MEDIUM_SIZE = 17
# plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes

from matplotlib import rcParams
# rcParams['text.usetex'] = False
rcParams['font.family'] = 'DejaVu Sans'
# rcParams['mathtext.fontset'] = 'custom'
# rcParams['mathtext.fontset'] = 'stix'
# rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


def quick2D(image, dx=None, title=None, logZ=False, vlim=(None,None), colormap=None):
    """
    Looks at a 2D array, has bunch of handles for plot.imshow

    :param image: 2D array to plot (data)
    :param dx: sampling of the image in m. Hardcoded to convert to um on axis
    :param title: string--must be set or will error!
    :param logZ: flag to set logscale plotting on z-axis
    :param vlim: tuple of limits on the colorbar axis, otherwise default matplotlib (pass in logscale limits if logZ=True)
    :param colormap: specify colormap as string
    :return:
    """
    # Create figure & adjust subplot number, layout, size, whitespace
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Title
    while title is None:
        print("Plots without titles: Don't Do It!")
        title = input("Please Enter Title: ")

    # X,Y lables
    if dx is not None:
        scale = np.round(
            np.linspace(-dx * sp.maskd_size / 2, dx * sp.maskd_size / 2, (sp.maskd_size + 1) / 50) * 1e6).astype(int)
        tic_spacing = np.linspace(0, sp.maskd_size, sp.maskd_size/50)
        tic_spacing[0] = tic_spacing[0] + 1  # hack for edge effects
        tic_spacing[-1] = tic_spacing[-1] -1  # hack for edge effects
        plt.xticks(tic_spacing, scale)
        plt.yticks(tic_spacing, scale)
        plt.xlabel('[um]')

    # Setting Logscale
    if logZ:
        if np.min(image) <= 0:
            cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vlim[0], vmax=vlim[1],
                            norm=LogNorm(),    #norm=SymLogNorm(linthresh=1e-5),
                            cmap="YlGnBu_r")
            clabel = "Log Normalized Intensity"
        else:
            cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vlim[0], vmax=vlim[1],
                            norm=LogNorm(), cmap="YlGnBu_r")
            clabel = "Log Normalized Intensity"
    else:
        cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vlim[0], vmax=vlim[1], cmap=colormap)
        clabel = "Normalized Intensity"

    # Plotting
    plt.title(title, fontweight='bold', fontsize=16)
    cb = plt.colorbar(cax)
    cb.set_label(clabel)
    plt.show(block=True)


def view_spectra(datacube, title=None, show=True, logZ=False, use_axis=True, vlim=(None,None), subplt_cols=3,
                  dx=None):
    """
    view plot of intensity in each wavelength bin at a single (last) timestep

    :param datacube: 3D spectral cube (n_wavelengths, nx, ny) at single timestep
    :param title: string, must be set or will error!
    :param show: flag possibly useful for plotting loops of things?
    :param logZ: turn logscale plotting for Z axis on or off
    :param use_axis: turn on/off using axis ticks, colorbar, etc
    :param vlim: tuple of colorbar axis limits (min,max)
    :param subplt_cols: number of subplots per row
    :param dx: sampling of the image in m. Hardcoded to convert to um
    :return:
    """
    plt.close('all')

    # Create figure & adjust subplot number, layout, size, whitespace
    fig = plt.figure()
    n_colors = len(datacube)
    n_rows = int(np.ceil(n_colors / float(subplt_cols))+1)
    plt.axis('off')
    gs = gridspec.GridSpec(n_rows, subplt_cols, wspace=0.08, top=0.9, bottom=0.2)

    # Title
    if title is None:
        title = input("Please Enter Title: ")
        pass
    fig.suptitle(title, fontweight='bold', fontsize=16)

    # Wavelength Strings for Subplot Titles
    w_string = np.array(np.linspace(ap.wvl_range[0] * 1e9, ap.wvl_range[1] * 1e9, ap.n_wvl_final, dtype=int), dtype=str)

    if dx is not None:
        dx = dx * 1e6  # [convert to um]

    for w in range(n_colors):
        ax = fig.add_subplot(gs[w])

        # X,Y lables
        if dx is not None:
            # dprint(f"sampling = {sampl[w]}")
            tic_spacing = np.linspace(0, sp.maskd_size, 5)  # 5 is just set by hand, arbitrarily chosen
            tic_lables = np.round(
                np.linspace(-dx * sp.maskd_size / 2, dx * sp.maskd_size / 2, 5)).astype(int)  # nsteps must be same as tic_spacing
            tic_spacing[0] = tic_spacing[0] + 1  # hack for edge effects
            tic_spacing[-1] = tic_spacing[-1] - 1  # hack for edge effects
            plt.xticks(tic_spacing, tic_lables, fontsize=6)
            plt.yticks(tic_spacing, tic_lables, fontsize=6)
            # plt.xlabel('[um]', fontsize=8)
            # plt.ylabel('[um]', fontsize=8)

        # Z-axis scale
        if logZ:
            if vlim[0] is not None and vlim[0] <= 0:
                ax.set_title(r'$\lambda$ = '+f"{w_string[w]} nm")
                im = ax.imshow(opx.extract_center(datacube[w]), interpolation='none', origin='lower',
                               vmin=vlim[0], vmax=vlim[1],
                               norm=SymLogNorm(linthresh=1e-5),
                               cmap="YlGnBu_r")
                clabel = "Log Normalized Intensity"
            else:
                ax.set_title(r'$\lambda$ = '+f"{w_string[w]} nm")
                im = ax.imshow(opx.extract_center(datacube[w]), interpolation='none', origin='lower',
                               vmin=vlim[0], vmax=vlim[1], norm=LogNorm(),
                               cmap="YlGnBu_r")
                clabel = "Log Normalized Intensity"
        else:
            ax.set_title(r'$\lambda$ = '+f"{w_string[w]} nm")
            im = ax.imshow(opx.extract_center(datacube[w]),
                           interpolation='none', origin='lower', vmin=vlim[0], vmax=vlim[1], cmap="YlGnBu_r")
            clabel = "Normalized Intensity"

        if use_axis == 'anno':
            annotate_axis(im, ax, datacube.shape[1])
        if use_axis is None:
            plt.axis('off')

    if use_axis:
        gs.tight_layout(fig, pad=0.08, rect=(0, 0, 1, 0.85))  # rect = (left, bottom, right, top)
        # fig.tight_layout(pad=50)
        cbar_ax = fig.add_axes([0.55, 0.1, 0.2, 0.05])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')  #
        cb.set_label(clabel)

    if show is True:
        plt.show(block=True)


def view_timeseries(img_tseries, title=None, show=True, logZ=False, use_axis=True, vlim =(None,None),
                    dx=None, subplt_cols=3):
    """
    view white light images in the timeseries

    :param img_tseries: complex timeseries
    :param title: string, must be set or will error!
    :param show: flag possibly useful for plotting loops of things?
    :param logZ: turn logscale plotting for Z-axis on or off
    :param use_axis: turn on/off using axis ticks, colorbar, etc
    :param vlim: tuple of colorbar axis limits (min,max)
    :param subplt_cols: number of subplots per row
    :param dx: sampling of the image in m. Hardcoded to convert to um
    :return:
    """
    plt.close('all')

    # Recreate CDI phase stream for plot titles
    if cdip.use_cdi:
        from medis.CDI import gen_CDI_phase_stream
        phases = gen_CDI_phase_stream()

    # Create figure & adjust subplot number, layout, size, whitespace
    fig = plt.figure()
    n_tsteps = len(img_tseries)
    n_rows = int(np.ceil(n_tsteps / float(subplt_cols))+1)
    plt.axis('off')
    gs = gridspec.GridSpec(n_rows, subplt_cols, wspace=0.08, top=0.9, bottom=0.2)

    # Title
    if title is None:
        raise NameError("Plots without titles: Don't Do It!")
        title = input("Please Enter Title: ")
        pass
    fig.suptitle(title, fontweight='bold', fontsize=16)

    # Converting Sampling Units to Readable numbers
    if dx[w] < 1e-6:
        dx[w] *= 1e6  # [convert to um]
        axlabel = 'um'
    elif dx[w] < 1e-3:
        dx[w] *= 1e3  # [convert to mm]
        axlabel = 'mm'
    elif 1e-2 > dx[w] > 1e-3:
        dx[w] *= 1e2  # [convert to cm]
        axlabel = 'cm'
    else:
        axlabel = 'm'

    # X,Y lables
    if dx is not None:
        # dprint(f"sampling = {sampl[w]}")
        tic_spacing = np.linspace(0, sp.maskd_size, 5)  # 5 (# of ticks) is just set by hand, arbitrarily chosen
        tic_lables = np.round(
            np.linspace(-dx[w] * sp.maskd_size / 2, dx[w] * sp.maskd_size / 2, 5)).astype(
            int)  # nsteps must be same as tic_spacing
        tic_spacing[0] = tic_spacing[0] + 1  # hack for edge effects
        tic_spacing[-1] = tic_spacing[-1] - 1  # hack for edge effects
        plt.xticks(tic_spacing, tic_lables, fontsize=6)
        plt.yticks(tic_spacing, tic_lables, fontsize=6)
        # plt.xlabel('[um]', fontsize=8)
        plt.ylabel(axlabel, fontsize=8)

    for t in range(n_tsteps):
        ax = fig.add_subplot(gs[t])
        if logZ:
            if vlim[0] is not None and vlim[0] <= 0:
                if cdip.use_cdi and not np.isnan(phases[t]):
                    ax.set_title(f"t={t * sp.sample_time}, CDI" r'$\theta$' + f"={phases[t]/np.pi:.2f}" + r'$\pi$')
                else:
                    ax.set_title(f"t={t*sp.sample_time}")
                im = ax.imshow(img_tseries[t], interpolation='none', origin='lower',
                               vmin=vlim[0], vmax=vlim[1],
                               norm=SymLogNorm(linthresh=1e-5), cmap="YlGnBu_r")
                clabel = "Log Normalized Intensity"
            else:
                if cdip.use_cdi and not np.isnan(phases[t]):
                    ax.set_title(f"t={t * sp.sample_time}, CDI" r'$\theta$' + f"={phases[t]/np.pi:.2f}" + r'$\pi$')
                else:
                    ax.set_title(f"t={t * sp.sample_time}")
                im = ax.imshow(img_tseries[t], interpolation='none', origin='lower',
                               vmin=vlim[0], vmax=vlim[1],
                               norm=LogNorm(), cmap="YlGnBu_r")
                clabel = "Log Normalized Intensity"
        else:
            if cdip.use_cdi and not np.isnan(phases[t]):
                ax.set_title(f"t={t * sp.sample_time}, CDI" r'$\theta$' + f"={phases[t]/np.pi:.2f}" + r'$\pi$')
            else:
                ax.set_title(f"t={t * sp.sample_time}")
            im = ax.imshow(img_tseries[t], interpolation='none', origin='lower',
                           vmin=vlim[0], vmax=vlim[1],
                           cmap="YlGnBu_r")
            clabel = "Normalized Intensity"

        if use_axis == 'anno':
            annotate_axis(im, ax, img_tseries.shape[1])
        if use_axis is None:
            plt.axis('off')

    if use_axis:
        gs.tight_layout(fig, pad=0.08, rect=(0, 0, 1, 0.85))  # rect = (left, bottom, right, top)
        # fig.tight_layout(pad=50)
        cbar_ax = fig.add_axes([0.55, 0.1, 0.2, 0.05])  # Add axes for colorbar @ position [left,bottom,width,height]
        cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')  #
        cb.set_label(clabel)

    if show is True:
        plt.show(block=True)


def plot_planes(cpx_seq, title=None, logZ=[False], use_axis=True, vlim=(None, None), subplt_cols=3,
                 dx=None):
    """
    view plot of intensity in each wavelength bin at a single (last) timestep
    will pull out the plane(s) of sp.save_list at last tstep of cpx_sequence, convert to intensity, and sum over
      wavelength and object

    :param cpx_seq:
    :param title: string, must be set or will error!
    :param logZ: turn logscale plotting for z-axis on or off
    :param use_axis: turn on/off using axis ticks, colorbar, etc
    :param vlim: tuple of colorbar axis limits (min,max)
    :param subplt_cols: number of subplots per row
    :param dx: sampling of the image at each saved plane
    :return:
    """
    plt.close('all')

    # Create figure & adjust subplot number, layout, size, whitespace
    fig = plt.figure()
    n_planes = len(sp.save_list)
    n_rows = int(np.ceil(n_planes / float(subplt_cols)) + 1)
    plt.axis('off')
    gs = gridspec.GridSpec(n_rows, subplt_cols, wspace=0.08, top=0.9, bottom=0.2)

    # Main Title
    if title is None:
        raise NameError("Plots without titles: Don't Do It!")
        title = input("Please Enter Title: ")
        pass
    fig.suptitle(title, fontweight='bold', fontsize=16)

    # Small Hack to repeat axis if defaults used
    if len(logZ) == 1:
        logZ = np.repeat(logZ,len(sp.save_list))
    if len(vlim) == 2:
        vlim = (vlim,)*len(sp.save_list)

    for w in range(n_planes):
        ax = fig.add_subplot(gs[w])

        # Retreiving Data
        plot_plane = sp.save_list[w]
        plane = opx.extract_plane(cpx_seq, plot_plane)
        mu.dprint(f"shape of plane in plot_plane is {plane.shape}")
        plane = np.sum(opx.cpx_to_intensity(plane[0]), axis=(0,1))
            # converts to intensity THEN sums over object, then sums over wavelength
        plane = opx.extract_center(plane)

        # Converting Sampling Units to Readable numbers
        if dx[w] < 1e-6:
            dx[w] *= 1e6  # [convert to um]
            axlabel = 'um'
        elif dx[w] < 1e-3:
            dx[w] *= 1e3  # [convert to mm]
            axlabel = 'mm'
        elif 1e-2 > dx[w] > 1e-3:
            dx[w] *= 1e2  # [convert to cm]
            axlabel = 'cm'
        else:
            axlabel = 'm'

        # X,Y lables
        if dx is not None:
            # dprint(f"sampling = {sampl[w]}")
            tic_spacing = np.linspace(0, sp.maskd_size, 5)  # 5 (# of ticks) is just set by hand, arbitrarily chosen
            tic_lables = np.round(
                np.linspace(-dx[w] * sp.maskd_size / 2, dx[w] * sp.maskd_size / 2, 5)).astype(int)  # nsteps must be same as tic_spacing
            tic_spacing[0] = tic_spacing[0] + 1  # hack for edge effects
            tic_spacing[-1] = tic_spacing[-1] - 1  # hack for edge effects
            plt.xticks(tic_spacing, tic_lables, fontsize=6)
            plt.yticks(tic_spacing, tic_lables, fontsize=6)
            # plt.xlabel('[um]', fontsize=8)
            plt.ylabel(axlabel, fontsize=8)

        # Z-axis scale
        if logZ[w]:
            if vlim[w][0] is not None and vlim[w][0] <= 0:
                ax.set_title(f"{sp.save_list[w]}")
                im = ax.imshow(plane, interpolation='none', origin='lower', vmin=vlim[w][0], vmax=vlim[w][1],
                               norm=SymLogNorm(linthresh=1e-5),
                               cmap="YlGnBu_r")
                cb = fig.colorbar(im)
                # clabel = "Log Normalized Intensity"
            else:
                ax.set_title(f"{sp.save_list[w]}")
                im = ax.imshow(plane, interpolation='none', origin='lower', vmin=vlim[w][0], vmax=vlim[w][1],
                               norm=LogNorm(), cmap="YlGnBu_r")  #
                cb = fig.colorbar(im)
                # clabel = "Log Normalized Intensity"
                # cb.set_label(clabel)
        else:
            ax.set_title(f"{sp.save_list[w]}")
            im = ax.imshow(plane, interpolation='none', origin='lower', vmin=vlim[w][0], vmax=vlim[w][1],
                           cmap="twilight")  # "YlGnBu_r"
            cb = fig.colorbar(im)  #
            # clabel = "Normalized Intensity"
            # cb.set_label(clabel)

    if use_axis:
        gs.tight_layout(fig, pad=0.08, rect=(0, 0, 1, 0.85))  # rect = (left, bottom, right, top)
        # fig.tight_layout(pad=50)

    plt.show(block=True)


# def grid(datacube, nrows=2, logZ=False, axis=None, width=None, titles=None, ctitles=None, annos=None,
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
#             if logZ:
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
# def indep_images(datacube, logZ=False, axis=None, width=None, titles=None, annos=None, scale=1, vmins=None,
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
#         if logZ:
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
# def compare_images(datacube, logZ=False, axis=None, width=None, title=None, annos=None, scale=1, max_scale=0.05,
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
#         if logZ:
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
# def loop_frames(frames, axis=False, circles=None, logZ=False, vmin=None, vmax=None, show=True):
#     fig, ax = plt.subplots()
#     ax.set(title='Click to update the data')
#     dprint((vmin, vmax))
#     if logZ:
#         if np.min(frames) < 0:
#             im = ax.imshow(frames[0], norm=SymLogNorm(linthresh=1e-3), origin='lower', vmin=vmin, vmax=vmax)
#         else:
#             im = ax.imshow(frames[0], norm=LogNorm(), origin='lower', vmin=vmin, vmax=vmax)
#     else:
#         im = ax.imshow(frames[0], origin='lower', vmin=vmin, vmax=vmax)
#     if axis:
#         annotate_axis(im, ax, frames.shape[1])
#     cbar = plt.colorbar(im)
#     # if logZ:
#     #     cbar_ticks = np.logspace(np.min(frames), np.max(frames), num=7, endpoint=True)
#     if not logZ:
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
#                 # if logZ:
#                 #     cbar_ticks = np.logspace(np.min(frames[self.f]), np.max(frames[self.f]), num=7, endpoint=True)
#                 # if not logZ:
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
