from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.table import Table
import numpy as np
import healpy as hp
import os


class MapperKiDS1000(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'data_catalog': 'KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits',
           'file_nz': SOM_N_of_Z/K1000_..._TOMO1_Nz.asc
          'zbin':0,
          'nside':nside,
          'mask_name': 'mask_KiDS1000_0',
          'path_lite': path}
        """

        self._get_defaults(config)
        self.path_lite = config.get('path_lite', None)
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
        # Values from Table 1 of 2007.01845
        self.m = (-0.009, -0.011, -0.015, 0.002, 0.007)

        self.cat_data = None
        self.w2s2 = None
        self.w2s2s = {'PSF': None, 'shear': None, 'stars': None}

        self.dndz = None
        self.sel = {'galaxies': 1, 'stars': 0}

        self.signal_map = None
        self.maps = {'PSF': None, 'shear': None, 'stars': None}

        self.mask = None
        self.masks = {'stars': None, 'galaxies': None}

        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None, 'stars': None}

        self.e1_flag = config.get('e1_flag', 'e1')
        self.e2_flag = config.get('e2_flag', 'e2')
        self.column_names = ['SG_FLAG', 'Z_B', 'Z_B_MIN', 'Z_B_MAX',
                             'ALPHA_J2000', 'DELTA_J2000', 'PSF_e1', 'PSF_e2',
                             self.e1_flag, self.e2_flag, 'weight']

    def get_catalog(self):
        if self.cat_data is None:
            read_lite, fname_lite = self._check_lite_exists(self.zbin, 'cat')
            if read_lite:
                print(f'loading lite cat {self.zbin}', flush=True)
                self.cat_data = Table.read(fname_lite, format='fits')
            else:
                print(f'loading full cat {self.zbin}', flush=True)
                self.cat_data = self._load_catalog()

            print('Catalogs loaded', flush=True)

        return self.cat_data

    def _load_catalog(self):
        nzbins = self.zbin_edges.shape[0]
        data = []

        data_cat = Table.read(self.config['data_catalog'],
                              format='fits')[self.column_names]
        for i in range(nzbins):
            # binning
            sel = self._bin_z(data_cat, i)
            data_zbin = data_cat[sel]
            self._remove_additive_bias(data_zbin)
            data.append(data_zbin)

        for zbin, cat_zbin in enumerate(data):
            self._remove_multiplicative_bias(cat_zbin, zbin)
            read_lite, fname_lite = self._check_lite_exists(zbin, 'cat')
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
            e1_flag = self.e1_flag
            e2_flag = self.e2_flag
        elif mode == 'PSF':
            kind = 'galaxies'
            e1_flag = 'PSF_e1'
            e2_flag = 'PSF_e2'
        elif mode == 'stars':
            kind = 'stars'
            e1_flag = self.e1_flag
            e2_flag = self.e2_flag
        else:
            raise ValueError(f"Unknown mode {mode}")
        return kind, e1_flag, e2_flag, mode

    def _check_lite_exists(self, zbin, suffix, gzip=False):
        if self.path_lite is None:
            return False, None
        else:
            fname_lite = self.path_lite + \
                f'KiDS1000_lite_{suffix}_zbin{zbin}.fits'
            if gzip:
                fname_lite += '.gz'
            return os.path.isfile(fname_lite), fname_lite

    def _bin_z(self, cat, zbin):
        z_key = 'Z_B'
        z_edges = self.zbin_edges[zbin]
        return ((cat[z_key] > z_edges[0]) &
                (cat[z_key] <= z_edges[1]))

    def _remove_additive_bias(self, cat):
        sel_gals = cat['SG_FLAG'] == 1
        if np.any(sel_gals):
            e1mean = np.average(cat[self.e1_flag][sel_gals],
                                weights=cat['weight'][sel_gals])
            e2mean = np.average(cat[self.e2_flag][sel_gals],
                                weights=cat['weight'][sel_gals])
            cat[self.e1_flag][sel_gals] -= e1mean
            cat[self.e2_flag][sel_gals] -= e2mean

    def _remove_multiplicative_bias(self, cat, zbin):
        sel_gals = cat['SG_FLAG'] == 1
        cat[self.e1_flag][sel_gals] /= 1 + self.m[zbin]
        cat[self.e2_flag][sel_gals] /= 1 + self.m[zbin]

    def _get_gals_or_stars(self, kind='galaxies'):
        cat_data = self.get_catalog()
        sel = cat_data['SG_FLAG'] == self.sel[kind]
        return cat_data[sel]

    def get_signal_map(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.maps[mod] is not None:
            self.signal_map = self.maps[mod]
            return self.signal_map

        # This will only be computed if self.maps['mod'] is None
        lite1, fname1 = self._check_lite_exists(self.zbin,
                                                f'{mod}_e1_ns{self.nside}',
                                                True)
        lite2, fname2 = self._check_lite_exists(self.zbin,
                                                f'{mod}_e2_ns{self.nside}',
                                                True)
        if lite1 and lite2:
            print('Loading bin{} signal map'.format(self.zbin))
            e1 = hp.read_map(fname1)
            e2 = hp.read_map(fname2)
            self.maps[mod] = [-e1, e2]
        else:
            print('Computing bin{} signal map'.format(self.zbin))
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

            # overwrite = True in case it is also being computed by other
            # process
            if fname1:
                hp.write_map(fname1, we1, overwrite=True)
            if fname2:
                hp.write_map(fname2, we2, overwrite=True)

            self.maps[mod] = [-we1, we2]

        self.signal_map = self.maps[mod]
        return self.signal_map

    def get_mask(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.masks[kind] is not None:
            self.mask = self.masks[kind]
            return self.mask
        lite, fn_lite = self._check_lite_exists(self.zbin,
                                                f'{kind}_mask_ns{self.nside}',
                                                True)
        if lite:
            print('Loading bin{} mask'.format(self.zbin))
            self.masks[kind] = hp.read_map(fn_lite)
        else:
            data = self._get_gals_or_stars(kind)
            self.masks[kind] = get_map_from_points(data, self.nside,
                                                   w=data['weight'],
                                                   ra_name='ALPHA_J2000',
                                                   dec_name='DELTA_J2000')
            if fn_lite:
                hp.write_map(fn_lite, self.masks[kind], overwrite=True)
        self.mask = self.masks[kind]
        return self.mask

    def _get_w2s2(self, mode):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.w2s2s[mod] is not None:
            self.w2s2 = self.w2s2s[mod]
            return self.w2s2
        lite, fn_lite = self._check_lite_exists(self.zbin,
                                                f'{kind}_w2s2_ns{self.nside}',
                                                True)
        if lite:
            print('Loading bin{} w2s2'.format(self.zbin))
            self.w2s2s[mod] = hp.read_map(fn_lite)
        else:
            data = self._get_gals_or_stars(kind)
            wcol = data['weight']**2*0.5*(data[e1f]**2+data[e2f]**2)
            self.w2s2s[mod] = get_map_from_points(data, self.nside, w=wcol,
                                                  ra_name='ALPHA_J2000',
                                                  dec_name='DELTA_J2000')
            if fn_lite:
                hp.write_map(fn_lite, self.w2s2s[mod], overwrite=True)
        self.w2s2 = self.w2s2s[mod]
        return self.w2s2

    def get_nl_coupled(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.nls[mod] is None:
            self.w2s2 = self._get_w2s2(mode)
            N_ell = hp.nside2pixarea(self.nside) * np.mean(self.w2s2)
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # ylm = 0 for l < spin
            self.nls[mod] = np.array([nl, 0*nl, 0*nl, nl])
        self.nl_coupled = self.nls[mod]
        return self.nl_coupled

    def get_nz(self, dz=0):
        if self.dndz is None:
            self.dndz = np.loadtxt(self.config['file_nz'], unpack=True)[:2]

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
