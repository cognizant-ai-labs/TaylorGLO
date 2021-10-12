import math
import argparse
import h5py
import numpy as np
from scipy import interpolate

import os

def h5_to_mathematica(surf_file, surf_name='train_loss', zmax=-1):
    #set this to True to generate points
    show_points = False
    #set this to True to generate polygons
    show_polys = True

    f = h5py.File(surf_file,'r')

    [xcoordinates, ycoordinates] = np.meshgrid(f['xcoordinates'][:], f['ycoordinates'][:][:])
    vals = f[surf_name]

    # DeNaNinize.
    vals = np.nan_to_num(vals, nan=zmax)

    vals = np.clip(vals, None, zmax)

    # for x in vals:
        # print(','.join([str(i) for i in x]))

    head, _ = os.path.split(surf_file)
    np.savetxt(head + "/grid_" + surf_name + '.csv', vals, delimiter=',')
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert h5 file to Mathematica')
    parser.add_argument('--surf_file', '-f', default='', help='The h5 file that contains surface values')
    parser.add_argument('--surf_name', default='train_loss',
		help='The type of surface to plot: train_loss | test_loss | train_acc | test_acc ')
    parser.add_argument('--zmax', default=-1, type=float, help='Maximum z value to map')
    args = parser.parse_args()

    h5_to_mathematica(args.surf_file, args.surf_name, zmax=args.zmax)
