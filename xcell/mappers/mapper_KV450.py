from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
import os


class MapperKV450(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'data_catalogs': ['KV450_G12_reweight_3x4x4_v2_good.cat',
                             'KV450_G23_reweight_3x4x4_v2_good.cat',
                             'KV450_GS_reweight_3x4x4_v2_good.cat',
                             'KV450_G15_reweight_3x4x4_v2_good.cat',
                             'KV450_G9_reweight_3x4x4_v2_good.cat'] ,
          'file_nz': Nz_DIR_z0.1t0.3.asc,
          'zbin':0,
          'nside':nside,
          'mask_name': 'mask_KV450_0',
          'path_lite': path}
        """

        self._get_defaults(config)
        self.path_lite = config.get('path_lite', None)
        self.column_names = ['SG_FLAG', 'GAAP_Flag_ugriZYJHKs',
                             'Z_B', 'Z_B_MIN', 'Z_B_MAX',
                             'ALPHA_J2000', 'DELTA_J2000', 'PSF_e1', 'PSF_e2',
                             'bias_corrected_e1', 'bias_corrected_e2',
                             'weight']

        self.mode = config.get('mode', 'shear')
        self.zbin_edges = np.array([[0.1, 0.3],
                                   [0.3, 0.5],
                                   [0.5, 0.7],
                                   [0.7, 0.9],
                                   [0.9, 1.2]])
        self.npix = hp.nside2npix(self.nside)
        self.zbin = int(config['zbin'])
        self.z_edges = self.zbin_edges[self.zbin]
        # Multiplicative bias
        # Values from Table 2 of 1812.06076 (KV450 cosmo paper)
        self.m = (-0.017, -0.008, -0.015, 0.010, 0.006)

        read_lite, fname_lite = self._check_lite_exists(self.zbin)
        if read_lite:
            print(f'loading lite cat {self.zbin}', flush=True)
            self.cat = Table.read(fname_lite, format='fits')
        else:
            print(f'loading full cat {self.zbin}', flush=True)
            self.cat = self._create_catalog()

        print('Catalogs loaded', flush=True)

        self.dndz = np.loadtxt(self.config['file_nz'], unpack=True)
        self.sel = {'galaxies': 1, 'stars': 0}

        self.signal_map = None
        self.maps = {'PSF': None, 'shear': None, 'stars': None}

        self.mask = None
        self.masks = {'stars': None, 'galaxies': None}

        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None, 'stars': None}

    def _create_catalog(self):
        nzbins = self.zbin_edges.shape[0]
        data = [Table() for i in range(nzbins)]

        for file_data in self.config['data_catalogs']:
            data_cat = Table.read(file_data, format='fits')[self.column_names]
            # GAAP cut
            goodgaap = data_cat['GAAP_Flag_ugriZYJHKs'] == 0
            data_cat = data_cat[goodgaap]

            for i in range(nzbins):
                # Binning
                sel = self._bin_z(data_cat, i)
                data_zbin = data_cat[sel]
                self._remove_additive_bias(data_zbin)
                data[i] = vstack([data[i], data_zbin])

        for zbin, cat_zbin in enumerate(data):
            self._remove_multiplicative_bias(cat_zbin, zbin)
            read_lite, fname_lite = self._check_lite_exists(zbin)
            print(fname_lite)
            if fname_lite is not None:
                print(f'Writing lite catalog: {fname_lite}', flush=True)
                cat_zbin.write(fname_lite)

        return data[self.zbin]

    def _set_mode(self, mode=None):
        if mode is None:
            mode = self.mode

        if mode == 'shear':
            kind = 'galaxies'
            e1_flag = 'bias_corrected_e1'
            e2_flag = 'bias_corrected_e2'
        elif mode == 'PSF':
            kind = 'galaxies'
            e1_flag = 'PSF_e1'
            e2_flag = 'PSF_e2'
        elif mode == 'stars':
            kind = 'stars'
            e1_flag = 'bias_corrected_e1'
            e2_flag = 'bias_corrected_e2'
        else:
            raise ValueError(f"Unknown mode {mode}")
        return kind, e1_flag, e2_flag, mode

    def _check_lite_exists(self, zbin):
        if self.path_lite is None:
            return False, None
        else:
            fname_lite = self.path_lite + f'KV450_lite_cat_zbin{zbin}.fits'
            return os.path.isfile(fname_lite), fname_lite

    def _bin_z(self, cat, zbin):
        z_key = 'Z_B'
        z_edges = self.zbin_edges[zbin]
        return ((cat[z_key] > z_edges[0]) &
                (cat[z_key] <= z_edges[1]))

    def _remove_additive_bias(self, cat):
        sel_gals = cat['SG_FLAG'] == 1
        if np.any(sel_gals):
            e1mean = np.average(cat['bias_corrected_e1'][sel_gals],
                                weights=cat['weight'][sel_gals])
            e2mean = np.average(cat['bias_corrected_e2'][sel_gals],
                                weights=cat['weight'][sel_gals])
            cat['bias_corrected_e1'][sel_gals] -= e1mean
            cat['bias_corrected_e2'][sel_gals] -= e2mean

    def _remove_multiplicative_bias(self, cat_data, zbin):
        sel_gals = cat_data['SG_FLAG'] == 1
        cat_data['bias_corrected_e1'][sel_gals] /= 1 + self.m[zbin]
        cat_data['bias_corrected_e2'][sel_gals] /= 1 + self.m[zbin]

    def _get_gals_or_stars(self, kind='galaxies'):
        sel = self.cat['SG_FLAG'] == self.sel[kind]
        return self.cat[sel]

    def get_signal_map(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.maps[mod] is None:
            data = self._get_gals_or_stars(kind)
            wcol = data['weight']*data[e1f]
            we1 = get_map_from_points(data, self.nside, w=wcol,
                                      ra_name='ALPHA_J2000',
                                      dec_name='DELTA_J2000')
            wcol = data['weight']*data[e2f]
            we2 = get_map_from_points(data, self.nside, w=wcol,
                                      ra_name='ALPHA_J2000',
                                      dec_name='DELTA_J2000')
            mask = self.get_mask(mod)
            goodpix = mask > 0
            we1[goodpix] /= mask[goodpix]
            we2[goodpix] /= mask[goodpix]
            self.maps[mod] = [-we1, we2]

        self.signal_map = self.maps[mod]
        return self.signal_map

    def get_mask(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.masks[kind] is None:
            data = self._get_gals_or_stars(kind)
            self.masks[kind] = get_map_from_points(data, self.nside,
                                                   w=data['weight'],
                                                   ra_name='ALPHA_J2000',
                                                   dec_name='DELTA_J2000')
        self.mask = self.masks[kind]
        return self.mask

    def get_nl_coupled(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.nls[mod] is None:
            data = self._get_gals_or_stars(kind)
            wcol = data['weight']**2*0.5*(data[e1f]**2+data[e2f]**2)
            w2s2 = get_map_from_points(data, self.nside, w=wcol,
                                       ra_name='ALPHA_J2000',
                                       dec_name='DELTA_J2000')
            N_ell = hp.nside2pixarea(self.nside) * np.mean(w2s2)
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # ylm = 0 for l < spin
            self.nls[mod] = np.array([nl, 0*nl, 0*nl, nl])
        self.nl_coupled = self.nls[mod]
        return self.nl_coupled

    def get_nz(self, dz=0):
        if not dz:
            return self.dndz

        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0

        return np.array([z_dz[sel], nz[sel]])

    def get_dtype(self):
        return 'galaxy_shear'

    def get_spin(self):
        return 2
