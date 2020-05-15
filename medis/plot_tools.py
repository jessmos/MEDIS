"""Quick plotting tools go here

Note: older code from Rupert used pylab, which is now discouraged. KD changed to pyplot on 7-10-19 but has
not tested all older aspects of the code to ensure proper switch from pylab (though should be the same)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
import warnings

from medis.params import tp, sp, iop, ap, cdip
from medis.utils import dprint
import medis.optics as opx
from medis.twilight_colormaps import sunlight

# MEDIUM_SIZE = 17
# plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes

from matplotlib import rcParams
# rcParams['text.usetex'] = False
rcParams['font.family'] = 'DejaVu Sans'
# rcParams['mathtext.fontset'] = 'custom'
# rcParams['mathtext.fontset'] = 'stix'
# rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


def quick2D(image, dx=None, title=None, logZ=False, vlim=(None,None), colormap=None, show=True):
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
    if colormap=='sunlight':
        colormap = sunlight
    # Create figure & adjust subplot number, layout, size, whitespace
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Title
    while title is None:
        warnings.warn("Plots without titles: Don't Do It!")
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

    if vlim == (None, None):
        nstd = 2
        std = np.std(image)
        mean = np.mean(image)
        vlim = mean - nstd * std, mean + nstd * std

    # Setting Logscale
    norm = None if not logZ else (LogNorm() if vlim[0] > 0 else SymLogNorm(1e-7))
    # if logZ:
        # if np.min(image) <= 0:
    cax = ax.imshow(image, interpolation='none', origin='lower', vmin=vlim[0], vmax=vlim[1],
                    norm=norm, cmap=colormap)
    clabel = "Intensity"

    # Plotting
    plt.title(title, fontweight='bold', fontsize=16)
    cb = plt.colorbar(cax)
    cb.set_label(clabel)
    if show:
        plt.show(block=True)

def grid(fields, title='body spectra', logZ=False, show=True, nstd=1, vlim=(None, None), cmap='inferno'):
    """
    General purpose plotter for multi-dimensional input tensors from 2D up to 6D. The tensor will be converted to 4D
    and plot as a grid of 2D images

    :param fields:
    :param title:
    :param logZ:
    :param show:
    :return:
    """
    if cmap == 'sunlight':
        cmap = sunlight
    fields = np.array(fields)  # just in case its a list
    assert fields.ndim > 2
    if np.iscomplexobj(fields.flat[0]):
        fields = np.abs(fields)**2  # convert to intensity if complex
    while len(fields.shape) > 4:
        try:
            boring_ind = fields.shape.index(1)
            fields = np.mean(fields, axis=boring_ind)
        except ValueError:
            fields = fields[0]  # if none are zero slice out first dimension
    while len(fields.shape) < 4:
        fields = fields[:,np.newaxis]

    slices = np.int_(np.ceil(np.array(fields.shape)[:2]/5))
    fields = fields[::slices[0],::slices[1]]
    print(f'fields being sliced by {slices} making new fields size {fields.shape}')
    nwave, nobj, x, y = fields.shape

    try:
        std = np.std(fields)
        mean = np.mean(fields)
    except ValueError:
        std = np.std(fields[0])
        mean = np.mean(fields[0])

    if vlim == (None, None):
        vmin, vmax = mean - nstd*std, mean + nstd*std
    else:
        vmin, vmax = vlim

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(title)
    norm = None if not logZ else (LogNorm() if vmin > 0 else SymLogNorm(1e-7))

    imgrid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(nobj, nwave),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.15,
                     )
    for i, ax in enumerate(imgrid):
        x, y = i % nwave, i // nwave
        im = ax.imshow(fields[x, y], norm=norm, vmin=vmin, vmax=vmax, cmap=cmap)

    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)

    if show:
        plt.show(block=True)

def view_spectra(datacube, title=None, show=True, logZ=False, use_axis=True, vlim=(None,None), subplt_cols=3,
                  dx=None, extract_center=False):
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
    gs = gridspec.GridSpec(n_rows, subplt_cols, wspace=0.08, top=0.9)

    # Title
    if title is None:
        warnings.warn("Plots without titles: Don't Do It!")
        # title = input("Please Enter Title: ")
        pass
    fig.suptitle(title, fontweight='bold', fontsize=16)

    # Wavelength Strings for Subplot Titles
    w_string = np.array(np.linspace(ap.wvl_range[0] * 1e9, ap.wvl_range[1] * 1e9, ap.n_wvl_final, dtype=int), dtype=str)

    for w in range(n_colors):
        ax = fig.add_subplot(gs[w])
        ax.set_title(r'$\lambda$ = ' + f"{w_string[w]} nm")

        # X,Y lables
        if dx is not None:
            dx[w] = dx[w] * 1e6  # [convert to um]
            # dprint(f"sampling = {sampl[w]}")
            tic_spacing = np.linspace(0, sp.maskd_size, 5)  # 5 (number of ticks) is set by hand, arbitrarily chosen
            tic_lables = np.round(
                np.linspace(-dx[w] * sp.maskd_size / 2, dx[w] * sp.maskd_size / 2, 5)).astype(int)  # nsteps must be same as tic_spacing
            tic_spacing[0] = tic_spacing[0] + 1  # hack for edge effects
            tic_spacing[-1] = tic_spacing[-1] - 1  # hack for edge effects
            plt.xticks(tic_spacing, tic_lables, fontsize=6)
            plt.yticks(tic_spacing, tic_lables, fontsize=6)
            # plt.xlabel('[um]', fontsize=8)
            # plt.ylabel('[um]', fontsize=8)

        if extract_center:
            slice = opx.extract_center(datacube[w])
        else:
            slice = datacube[w]

        # Z-axis scale
        if logZ:
            if np.min(slice) < 0:
                im = ax.imshow(slice, interpolation='none', origin='lower',
                               vmin=vlim[0], vmax=vlim[1],
                               norm=SymLogNorm(linthresh=1e-5),
                               cmap="YlGnBu_r")
                clabel = "Log Normalized Intensity"
            else:
                im = ax.imshow(slice, interpolation='none', origin='lower',
                               vmin=vlim[0], vmax=vlim[1], norm=LogNorm(),
                               cmap="YlGnBu_r")
                clabel = "Log Normalized Intensity"
        else:
            im = ax.imshow(slice,
                           interpolation='none', origin='lower', vmin=vlim[0], vmax=vlim[1], cmap="YlGnBu_r")
            clabel = "Normalized Intensity"

        if use_axis == 'anno':
            ax.annotate_axis(im, ax, datacube.shape[1])
        if use_axis is None:
            plt.axis('off')

    if use_axis:
        warnings.simplefilter("ignore", category=UserWarning)
        gs.tight_layout(fig, pad=1.08, rect=(0, 0, 1, 0.85))  # rect = (left, bottom, right, top)
        # fig.tight_layout(pad=50)
        cbar_ax = fig.add_axes([0.55, 0.3, 0.2, 0.05])  # Add axes for colorbar @ position [left,bottom,width,height]
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
    gs = gridspec.GridSpec(n_rows, subplt_cols, wspace=0.08, top=0.9, bottom=0.02)

    # Title
    if title is None:
        warnings.warn("Plots without titles: Don't Do It!")
        title = input("Please Enter Title: ")
        pass
    fig.suptitle(title, fontweight='bold', fontsize=16)

    for t in range(n_tsteps):
        ax = fig.add_subplot(gs[t])

        # Converting Sampling Units to Readable numbers
        if dx[t] < 1e-6:
            dx[t] *= 1e6  # [convert to um]
            axlabel = 'um'
        elif dx[t] < 1e-3:
            dx[t] *= 1e3  # [convert to mm]
            axlabel = 'mm'
        elif 1e-2 > dx[t] > 1e-3:
            dx[t] *= 1e2  # [convert to cm]
            axlabel = 'cm'
        else:
            axlabel = 'm'

        # X,Y lables
        if dx is not None:
            # dprint(f"sampling = {sampl[w]}")
            tic_spacing = np.linspace(0, sp.maskd_size, 5)  # 5 (# of ticks) is just set by hand, arbitrarily chosen
            tic_lables = np.round(
                np.linspace(-dx[t] * sp.maskd_size / 2, dx[t] * sp.maskd_size / 2, 5)).astype(
                int)  # nsteps must be same as tic_spacing
            tic_spacing[0] = tic_spacing[0] + 1  # hack for edge effects
            tic_spacing[-1] = tic_spacing[-1] - 1  # hack for edge effects
            plt.xticks(tic_spacing, tic_lables, fontsize=6)
            plt.yticks(tic_spacing, tic_lables, fontsize=6)
            # plt.xlabel('[um]', fontsize=8)
            plt.ylabel(axlabel, fontsize=8)

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
            ax.annotate_axis(im, ax, img_tseries.shape[1])
        if use_axis is None:
            plt.axis('off')

    if use_axis:
        warnings.simplefilter("ignore", category=UserWarning)
        gs.tight_layout(fig, pad=1.08, rect=(0, 0.02, 1, 0.85))  # rect = (left, bottom, right, top)
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

    Currently, the atmosphere and enterance pupil are plotted in units of phase vs intensity. I think we should change
    this later for user-specification

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
    n_rows = int(np.ceil(n_planes / float(subplt_cols)) )
    plt.axis('off')
    gs = gridspec.GridSpec(n_rows, subplt_cols, wspace=0.08)

    # Main Title
    if title is None:
        warnings.warn("Plots without titles: Don't Do It!")
        title = input("Please Enter Title: ")
        pass
    fig.suptitle(title, fontweight='bold', fontsize=16)

    # Small Hack to repeat axis if defaults used
    if len(logZ) == 1:
        logZ = np.repeat(logZ,len(sp.save_list))
    if len(vlim) == 2:
        vlim = (vlim,)*len(sp.save_list)

    for p in range(n_planes):
        ax = fig.add_subplot(gs[p])

        ###################
        # Retreiving Data
        ##################
        # Standard-Way
        # [timestep, plane, wavelength, object, x, y]
        plot_plane = sp.save_list[p]
        plane = opx.extract_plane(cpx_seq, plot_plane)
        # converts to intensity of last timestep, THEN sums over wavelength, then sums over object
        if plot_plane == "atmosphere" or plot_plane == "entrance_pupil":
            plane = np.sum(np.angle(plane[-1]), axis=(0,1))
            logZ[p] = False
            vlim[p] = (None,None)
            phs = " Phase"
        else:
            plane = np.sum(opx.cpx_to_intensity(plane[-1]), axis=(0,1))
            phs = ""
        plane = opx.extract_center(plane)
        ### Retreiving Data- Custom selection of plane ###
        # plot_plane = sp.save_list[w]
        # plane = opx.extract_plane(cpx_seq, plot_plane)
        # # plane = opx.cpx_to_intensity(plane[-1])
        # dprint(f"plane shape is {plane.shape}")
        # plane = opx.extract_center(np.angle(plane[0,1,1]))  # wavelengths, objects
        # dprint(f"plane shape is {plane.shape}")

        # Converting Sampling Units to Readable numbers
        if dx[p,0] < 1e-6:
            dx[p,:] *= 1e6  # [convert to um]
            axlabel = 'um'
        elif dx[p,0] < 1e-3:
            dx[p,:] *= 1e3  # [convert to mm]
            axlabel = 'mm'
        elif 1e-2 > dx[p,0] > 1e-3:
            dx[p,:] *= 1e2  # [convert to cm]
            axlabel = 'cm'
        else:
            axlabel = 'm'

        # X,Y lables
        if dx is not None:
            # dprint(f"sampling = {sampl[w]}")
            tic_spacing = np.linspace(0, sp.maskd_size, 5)  # 5 (# of ticks) is just set by hand, arbitrarily chosen
            tic_lables = np.round(
                np.linspace(-dx[p,0] * sp.maskd_size / 2, dx[p,0] * sp.maskd_size / 2, 5)).astype(int)  # nsteps must be same as tic_spacing
            tic_spacing[0] = tic_spacing[0] + 1  # hack for edge effects
            tic_spacing[-1] = tic_spacing[-1] - 1  # hack for edge effects
            plt.xticks(tic_spacing, tic_lables, fontsize=6)
            plt.yticks(tic_spacing, tic_lables, fontsize=6)
            # plt.xlabel('[um]', fontsize=8)
            plt.ylabel(axlabel, fontsize=8)

        # Z-axis scale
        if logZ[p]:
            if vlim[p][0] is not None and vlim[p][0] <= 0:
                ax.set_title(f"{sp.save_list[p]}"+phs)
                im = ax.imshow(plane, interpolation='none', origin='lower', vmin=vlim[p][0], vmax=vlim[p][1],
                               norm=SymLogNorm(linthresh=1e-5),
                               cmap="YlGnBu_r")
                cb = fig.colorbar(im)
                # clabel = "Log Normalized Intensity"
            else:
                ax.set_title(f"{sp.save_list[p]}"+phs)
                im = ax.imshow(plane, interpolation='none', origin='lower', vmin=vlim[p][0], vmax=vlim[p][1],
                               norm=LogNorm(), cmap="YlGnBu_r")  #(1e-6,1e-3)
                cb = fig.colorbar(im)
                # clabel = "Log Normalized Intensity"
                # cb.set_label(clabel)
        else:
            ax.set_title(f"{sp.save_list[p]}"+phs)
            im = ax.imshow(plane, interpolation='none', origin='lower', vmin=vlim[p][0], vmax=vlim[p][1],
                           cmap="YlGnBu_r")  #  "twilight"
            cb = fig.colorbar(im)  #
            # clabel = "Normalized Intensity"
            # cb.set_label(clabel)

    if use_axis:
        warnings.simplefilter("ignore", category=UserWarning)
        gs.tight_layout(fig, pad=1.08, rect=(0, 0.02, 1, 0.9))  # rect = (left, bottom, right, top)
        # fig.tight_layout(pad=50)

    plt.show(block=True)
