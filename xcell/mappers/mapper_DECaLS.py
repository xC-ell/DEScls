from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.integrate import simps
import numpy as np
import healpy as hp
import pymaster as nmt
import os


class MapperDECaLS(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'data_catalogs':['Legacy_Survey_BASS-MZLS_galaxies-selection.fits'],
           'zbin': 0,
           'z_name': 'PHOTOZ_3DINFER',
           'z_arr_dim': 500,
           'binary_mask': 'Legacy_footprint_final_mask.fits',
           'nl_analytic': True,
           'completeness_map': 'Legacy_footprint_completeness_mask_128.fits',
           'star_map': 'allwise_total_rot_1024.fits',
           'nside': 1024,
           'mask_name': 'mask_DECaLS'}
        """
        self._get_defaults(config)
        self.pz = config.get('z_name', 'PHOTOZ_3DINFER')
        self.z_arr_dim = config.get('z_arr_dim', 500)

        self.cat_data = None
        self.npix = hp.nside2npix(self.nside)

        bin_edges = [[0.00, 0.30],
                     [0.30, 0.45],
                     [0.45, 0.60],
                     [0.60, 0.80]]

        self.zbin = config['zbin']
        self.z_edges = bin_edges[self.zbin]

        # Angular mask flag
        self.mskflag = None
        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None
        self.comp_map = None
        self.stars = None
        self.bmask = None

    def get_catalogs(self):
        if self.cat_data is None:
            self.cat_data = []
            for file_data in self.config['data_catalogs']:
                if not os.path.isfile(file_data):
                    raise ValueError(f"File {file_data} not found")
                with fits.open(file_data) as f:
                    self.cat_data.append(Table.read(f))
            self.cat_data = vstack(self.cat_data)
            self.cat_data = self._bin_z(self.cat_data)

        return self.cat_data

    def _get_angmask(self):
        if self.mskflag is None:
            cat = self.get_catalogs()
            bmask = hp.read_map(self.config['binary_mask'],
                                verbose=False)
            nside = hp.npix2nside(len(bmask))
            ipix = hp.ang2pix(nside, cat['RA'], cat['DEC'],
                              lonlat=True)
            self.mskflag = bmask[ipix] > 0.
        return self.mskflag

    def _bin_z(self, cat):
        return cat[(cat[self.pz] >= self.z_edges[0]) &
                   (cat[self.pz] < self.z_edges[1])]

    def get_nz(self, dz=0):
        if self.dndz is None:
            mskflag = self._get_angmask()
            h, b = np.histogram(self.cat_data[self.pz][mskflag],
                                range=[-0.3, 1], bins=self.z_arr_dim)
            z_arr = 0.5 * (b[:-1] + b[1:])
            kernel = self._get_lorentzian(z_arr)
            nz_photo = h.astype(float)
            nz_spec = simps(kernel*nz_photo[None, :], x=z_arr, axis=1)
            nz_spec /= simps(nz_spec, x=z_arr)
            self.dndz = np.array([z_arr, nz_spec])

        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0
        return np.array([z_dz[sel], nz[sel]])

    def _get_lorentzian(self, zz):
        # a m s
        params = np.array([[1.257, -0.0010, 0.0122],
                           [1.104, 0.0076, 0.0151],
                           [1.476, -0.0024, 0.0155],
                           [2.019, -0.0042, 0.0265]])
        [a, m, s] = params[self.zbin]
        return 1./(1+((zz[:, None]-zz[None, :]-m)/s)**2/(2*a))**a

    def get_signal_map(self, apply_galactic_correction=True):
        if self.delta_map is None:
            d = np.zeros(self.npix)
            cat_data = self.get_catalogs()
            self.comp_map = self._get_comp_map()
            self.bmask = self._get_binary_mask()
            self.stars = self._get_stars()
            nmap_data = get_map_from_points(cat_data, self.nside)
            mean_n = self._get_mean_n(nmap_data)
            goodpix = self.bmask > 0
            d[goodpix] = nmap_data[goodpix]/(mean_n*self.comp_map[goodpix])-1
            if apply_galactic_correction:
                gcorr = self._get_galactic_correction(d, self.stars,
                                                      self.bmask)
                d -= gcorr['delta_map']
            self.delta_map = d
        return [self.delta_map]

    def _get_galactic_correction(self, delta, stars, bmask, nbins=14, npoly=5):
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
        # Now create a 5-th order polynomial fitting these data
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

    def _get_mean_n(self, nmap):
        self.comp_map = self._get_comp_map()
        self.stars = self._get_stars()
        self.bmask = self._get_binary_mask()
        goodpix = self.comp_map > 0.95
        goodpix *= self.stars < 8515
        goodpix *= self.bmask > 0
        n_mean = np.sum(nmap[goodpix])/np.sum(self.comp_map[goodpix])
        return n_mean

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

    def _get_comp_map(self):
        if self.comp_map is None:
            self.comp_map = hp.ud_grade(hp.read_map(
                                        self.config['completeness_map'],
                                        verbose=False),
                                        nside_out=self.nside)
        return self.comp_map

    def _get_binary_mask(self):
        if self.bmask is None:
            self.bmask = hp.ud_grade(hp.read_map(self.config['binary_mask'],
                                                 verbose=False),
                                     nside_out=self.nside)
        return self.bmask

    def get_mask(self):
        if self.mask is None:
            self.bmask = self._get_binary_mask()
            self.comp_map = self._get_comp_map()
            self.mask = self.bmask * self.comp_map
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            if (self.nside < 4096) or (self.config.get('nl_analytic', True)):
                cat_data = self.get_catalogs()
                n = get_map_from_points(cat_data, self.nside)
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
