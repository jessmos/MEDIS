

import numpy as np
from medis.mm_params import ap, tp


################################################################################
# MKIDs related Interpolation Stuff
################################################################################
def arange_into_cube(packets, size):
    # print 'Sorting packets into xy grid (no phase or time sorting)'
    cube = [[[] for i in range(size[0])] for j in range(size[1])]
    for ip, p in enumerate(packets):
        x = np.int_(p[2])
        y = np.int_(p[3])
        cube[x][y].append([p[0] ,p[1]])
        if len(packets) >= 1e7 and i p %1000 0= =0:
            misc.progressBar(value=ip, endvalue=len(packets))
    # print cube[x][y]
    # cube = time_sort(cube)
    return cube

def get_packets(datacube, step, dp, mp):
    if (mp.array_size != datacube[0].shape + np.array([1 ,1])).all():
        left = int(np.floor(float(sp.grid_size -mp.array_size[0] ) /2))
        right = int(np.ceil(float(sp.grid_size -mp.array_size[0] ) /2))
        top = int(np.floor(float(sp.grid_size -mp.array_size[1] ) /2))
        bottom = int(np.ceil(float(sp.grid_size -mp.array_size[1] ) /2))

        dprint(f"left={left},right={right},top={top},bottom={bottom}")
        datacube = datacube[:, bottom:-top, left:-right]

    if mp.respons_var:
        datacube *= dp.response_map[:datacube.shape[1] ,:datacube.shape[1]]

    num_events = int(ap.star_photons * ap.exposure_time * np.sum(datacube))
    dprint(f"# events ={num_events}, star photons = {ap.star_photons}, "
           f"sum(datacube) = {np.sum(datacube),}, Exposure Time ={ap.exposure_time}")
    if num_events * sp.num_processes > 1.0e9:
        dprint(num_events)
        dprint('Possibly too many photons for memory. Are you sure you want to do this? Remove exit() if so')
        exit()

    photons = temp.sample_cube(datacube, num_events)
    photons = temp.assign_calibtime(photons, step)

    thresh = dp.basesDeg[np.int_(photons[3]) ,np.int_(photons[2])] < -1 * photons[1]
    photons = photons[:, thresh]

    packets = np.transpose(photons)

    dprint("Completed Readout Loop")

    return packets
