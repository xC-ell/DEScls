from .mapper_base import MapperBase
from .utils import get_map_from_points, get_DIR_Nz
from astropy.io import fits
from astropy.table import Table
import pymaster as nmt
import numpy as np
import healpy as hp
import os


class MapperWIxSC(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'data_catalog': 'WIxSC.fits',
           'mask': 'mask.fits',
           'star_map': 'stars.fits',
           'spec_sample': 'zSpec-comp-WIxSC.csv',
           'bin_name': '0',
           'z_edges': [0, 0.5],
           'path_rerun': '.',
           'n_jk_dir': 100,
           'mask_name': 'mask_WIxSC'}
        """
        self._get_defaults(config)
        self.z_edges = config.get('z_edges', [0, 0.5])

        self.cat_data = None
        self.npix = hp.nside2npix(self.nside)
        self.bn = self.config['bin_name']

        # Angular mask
        self.dndz = None
        self.mask = None
        self.stars = None
        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None

    def get_catalog(self):
        if self.cat_data is None:
            fn = 'WIxSC_lite_bin' + self.bn + '.fits'
            f_exists, fname_lite = self._check_rerun_file_exists(fn)
            # Check if lite catalog exists
            if f_exists:
                with fits.open(fname_lite) as f:
                    self.cat_data = Table.read(f, format='fits', memmap=True)
            else:
                file_data = self.config['data_catalog']
                if not os.path.isfile(file_data):
                    raise ValueError(f"File {file_data} not found")
                # Read catalog
                with fits.open(file_data) as f:
                    self.cat_data = Table.read(f, format='fits', memmap=True)
                # Bin in redshift
                self.cat_data = self._bin_z(self.cat_data)
                # Ditch useless columns
                self.cat_data.keep_columns(['RA', 'DEC',
                                            'W1MCORR', 'W2MCORR',
                                            'RCALCORR', 'BCALCORR'])
                # Sky mask
                self.cat_data = self._mask_catalog(self.cat_data)
                # Save lite if needed
                if fname_lite is not None:
                    self.cat_data.write(fname_lite)
        return self.cat_data

    def _check_rerun_file_exists(self, fname):
        path_i = self.config.get('path_rerun', None)
        if path_i is None:
            return False, None
        else:
            fname_full = os.path.join(path_i, fname)
            return os.path.isfile(fname_full), fname_full

    def _mask_catalog(self, cat):
        self.mask = self.get_mask()
        ipix = hp.ang2pix(self.nside,
                          np.degrees(cat['RA']),
                          np.degrees(cat['DEC']),
                          lonlat=True)
        return cat[self.mask[ipix] > 0.1]

    def _bin_z(self, cat):
        return cat[(cat['ZPHOTO_CORR'] >= self.z_edges[0]) &
                   (cat['ZPHOTO_CORR'] < self.z_edges[1])]

    def _get_specsample(self, cat):
        import pandas as pd
        # Read full spectroscopic sample
        ds = Table.from_pandas(pd.read_csv(self.config['spec_sample']))
        msk = ((ds['zCorr'] < self.z_edges[-1]) &
               (ds['zCorr'] >= self.z_edges[0]))
        return ds[msk]

    def get_nz(self, dz=0, return_jk_error=False):
        if self.dndz is None:
            fn = 'nz_WIxSC_bin' + self.bn + '.npz'
            f_exists, f_name = self._check_rerun_file_exists(fn)
            # Read from file if it exists
            if f_exists:
                d = np.load(f_name)
                zm = d['z_mid']
                nz = d['nz']
                nz_jk = d['nz_jk']
            else:  # Else compute DIR N(z) and jackknife resamples
                c_p = self.get_catalog()
                c_s = self._get_specsample(c_p)
                # Sort spec sample by nested pixel index so jackknife
                # samples are spatially correlated.
                ip_s = hp.ring2nest(self.nside,
                                    hp.ang2pix(self.nside, c_s['ra_WISE'],
                                               c_s['dec_WISE'], lonlat=True))
                idsort = np.argsort(ip_s)
                c_s = c_s[idsort]
                # Compute DIR N(z)
                z, nz, nz_jk = get_DIR_Nz(c_s, c_p,
                                          ['W1c', 'W2c', 'Bcc', 'Rcc'],
                                          zflag='Zspec',
                                          zrange=[0, 1.],
                                          nz=100,
                                          bands_photo=['W1MCORR', 'W2MCORR',
                                                       'BCALCORR', 'RCALCORR'],
                                          njk=self.config.get('n_jk_dir', 100))
                zm = 0.5*(z[1:] + z[:-1])
                if f_name is not None:
                    np.savez(f_name, z_mid=zm, nz=nz, nz_jk=nz_jk)
            self.dndz = (zm, nz, nz_jk)

        z, nz, nz_jk = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0
        if return_jk_error:
            njk = len(nz_jk)
            enz = np.std(nz_jk, axis=0)*np.sqrt((njk-1)**2/njk)
            return np.array([z_dz[sel], nz[sel], enz[sel]])
        else:
            return np.array([z_dz[sel], nz[sel]])

    def get_signal_map(self, apply_galactic_correction=True):
        if self.delta_map is None:
            d = np.zeros(self.npix)
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            self.stars = self._get_stars()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            ra_name='RA', dec_name='DEC',
                                            in_radians=True)
            mean_n = self._get_mean_n(nmap_data)
            goodpix = self.mask > 0
            # Division by mask not really necessary, since it's binary.
            d[goodpix] = nmap_data[goodpix]/(mean_n*self.mask[goodpix])-1
            if apply_galactic_correction:
                gcorr = self._get_galactic_correction(d, self.stars,
                                                      self.mask)
                d -= gcorr['delta_map']
            self.delta_map = d
        return [self.delta_map]

    def get_mask(self):
        if self.mask is None:
            self.mask = hp.ud_grade(hp.read_map(self.config['mask'],
                                                verbose=False),
                                    nside_out=self.nside)
        return self.mask

    def _get_stars(self):
        if self.stars is None:
            # Power = -2 makes sure the total number of stars is conserved
            self.stars = hp.ud_grade(hp.read_map(self.config['star_map'],
                                                 verbose=False),
                                     nside_out=self.nside, power=-2)
            # Convert to stars per deg^2
            pix_srad = 4*np.pi/self.npix
            pix_deg2 = pix_srad*(180/np.pi)**2
            self.stars /= pix_deg2
        return self.stars

    def _get_mean_n(self, nmap):
        self.stars = self._get_stars()
        self.mask = self.get_mask()
        goodpix = self.stars < 8515
        goodpix *= self.mask > 0
        n_mean = np.sum(nmap[goodpix])/np.sum(self.mask[goodpix])
        return n_mean

    def _get_galactic_correction(self, delta, stars, bmask, nbins=14, npoly=3):
        # Create bins of star density
        stbins = np.linspace(np.log10(stars[bmask > 0].min()),
                             np.log10(stars[bmask > 0].max()), nbins+1)
        d_mean = []
        d_std = []
        # Loop over bins and compute <delta> and std(delta) in each star bin
        for ib, st in enumerate(stbins[:-1]):
            stp1 = stbins[ib+1]
            msk = bmask > 0
            msk *= np.log10(stars) <= stp1
            msk *= np.log10(stars) >= st
            dm = np.mean(delta[msk])
            dn = np.sum(msk).astype(float)
            ds = np.std(delta[msk])/np.sqrt(dn)
            d_mean.append(dm)
            d_std.append(ds)
        d_mean = np.array(d_mean)
        d_std = np.array(d_std)
        # Now create a 3-rd order polynomial fitting these data
        stmid = 0.5*(stbins[1:]+stbins[:-1])
        params = np.polyfit(stmid, d_mean, npoly, w=1./d_std)
        df = np.poly1d(params)
        # Create correction map
        d_corr = np.zeros_like(delta)
        d_corr[bmask > 0] = df(np.log10(stars[bmask > 0]))
        return {'stars': stmid,
                'delta_mean': d_mean,
                'delta_std': d_std,
                'delta_func': df,
                'delta_map': d_corr}

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            if (self.nside < 4096) or (self.config.get('nl_analytic', True)):
                cat_data = self.get_catalog()
                n = get_map_from_points(cat_data, self.nside,
                                        ra_name='RA', dec_name='DEC',
                                        in_radians=True)
                N_mean = self._get_mean_n(n)
                N_mean_srad = N_mean * self.npix / (4 * np.pi)
                mask = self.get_mask()
                N_ell = np.mean(mask) / N_mean_srad
            else:
                f = self.get_nmt_field()
                cl = nmt.compute_coupled_cell(f, f)[0]
                N_ell = np.mean(cl[4000:2*self.nside])
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
