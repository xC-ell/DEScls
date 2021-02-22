from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
import pymaster as nmt
import os


class MappereBOSSQSO(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'data_catalogs':['eBOSS_QSO_clustering_data-NGC-vDR16.fits'],
           'random_catalogs':['eBOSS_QSO_clustering_random-NGC-vDR16.fits'],
           'z_edges':[0, 1.5],
           'nside':nside,
           'nside_mask': nside_mask,
           'mask_name': 'mask_QSO_NGC_1'}
        """
        self._get_defaults(config)

        self.cat_data = []
        self.cat_random = []
        self.z_arr_dim = config.get('z_arr_dim', 50)

        for file_data, file_random in zip(self.config['data_catalogs'],
                                          self.config['random_catalogs']):
            if not os.path.isfile(file_data):
                raise ValueError(f"File {file_data} not found")
            with fits.open(file_data) as f:
                self.cat_data.append(Table.read(f))
            if not os.path.isfile(file_random):
                raise ValueError(f"File {file_random} not found")
            with fits.open(file_random) as f:
                self.cat_random.append(Table.read(f))

        self.cat_data = vstack(self.cat_data)
        self.cat_random = vstack(self.cat_random)
        self.nside_mask = config.get('nside_mask', self.nside)
        self.npix = hp.nside2npix(self.nside)

        self.z_edges = config['z_edges']

        self.cat_data = self._bin_z(self.cat_data)
        self.cat_random = self._bin_z(self.cat_random)
        self.w_data = self._get_weights(self.cat_data)
        self.w_random = self._get_weights(self.cat_random)
        self.alpha = np.sum(self.w_data)/np.sum(self.w_random)

        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None

    def _bin_z(self, cat):
        return cat[(cat['Z'] >= self.z_edges[0]) &
                   (cat['Z'] < self.z_edges[1])]

    def _get_weights(self, cat):
        cat_SYSTOT = np.array(cat['WEIGHT_SYSTOT'])
        cat_CP = np.array(cat['WEIGHT_CP'])
        cat_NOZ = np.array(cat['WEIGHT_NOZ'])
        weights = cat_SYSTOT*cat_CP*cat_NOZ  # FKP left out
        return weights

    def get_nz(self, dz=0):
        if self.dndz is None:
            h, b = np.histogram(self.cat_data['Z'], bins=self.z_arr_dim,
                                weights=self.w_data, range=[0.5, 2.5])
            self.dndz = np.array([0.5 * (b[:-1] + b[1:]), h])

        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0

        return np.array([z_dz[sel], nz[sel]])

    def get_signal_map(self):
        if self.delta_map is None:
            self.delta_map = np.zeros(self.npix)
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            w=self.w_data)
            nmap_random = get_map_from_points(self.cat_random, self.nside,
                                              w=self.w_random)

            mask = self.get_mask()
            goodpix = mask > 0
            self.delta_map = (nmap_data - self.alpha * nmap_random)
            self.delta_map[goodpix] /= mask[goodpix]
        return [self.delta_map]

    def get_mask(self):
        if self.mask is None:
            self.mask = get_map_from_points(self.cat_random,
                                            self.nside_mask,
                                            w=self.w_random)
            self.mask *= self.alpha
            # Account for different pixel areas
            area_ratio = (self.nside_mask/self.nside)**2
            self.mask = area_ratio * hp.ud_grade(self.mask,
                                                 nside_out=self.nside)
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            if self.nside < 4096:
                print('calculing nl from weights')
                pixel_A = 4*np.pi/hp.nside2npix(self.nside)
                mask = self.get_mask()
                w2_data = get_map_from_points(self.cat_data, self.nside,
                                              w=self.w_data**2)
                w2_random = get_map_from_points(self.cat_random, self.nside,
                                                w=self.w_random**2)
                goodpix = mask > 0
                N_ell = (w2_data[goodpix].sum() +
                         self.alpha**2*w2_random[goodpix].sum())
                N_ell *= pixel_A**2/(4*np.pi)
                self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
            else:
                print('calculating nl from mean cl values')
                f = self.get_nmt_field()
                cl = nmt.compute_coupled_cell(f, f)[0]
                N_ell = np.mean(cl[2000:2*self.nside])
                self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
