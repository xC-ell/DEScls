from mapper_base import MapperBase
from astropy.io import fits
from astropy.table import Table

import pandas as pd
import numpy as np
import healpy as hp
import pymaster as nmt
import os

class MapperDES(MapperBase):
    def __init__(self, config):
        
        self.config = config
        
        for file_data, file_random, file_mask, file_nz in zip(self.config['data_catalogs'],
            self.config['random_catalogs'], self.config['file_mask'], self.config['nz_file']):
            
            if not os.path.isfile(file_data):
                raise ValueError(f"File {file_data} not found")
            with fits.open(file_data) as f:
                self.cat_data = Table.read(f).to_pandas()
            
            if not os.path.isfile(file_random):
                raise ValueError(f"File {file_random} not found")
            with fits.open(file_random) as f:
                self.cat_random = Table.read(f).to_pandas()
                
            if not os.path.isfile(file_mask):
                raise ValueError(f"File {file_mask} not found")
            with fits.open(file_mask) as f:
                self.mask = hp.read_map(f, verbose = False)
                #self.mask = Table.read(f, format='fits').to_pandas()
                
            if not os.path.isfile(file_nz):
                raise ValueError(f"File {file_nz} not found")
            with fits.open(file_nz) as f:
                self.nz = Table.read(f).to_pandas()
            
        self.nside = config['nside']
        self.nside_mask = config.get('nside_mask', self.nside) 
        self.npix = hp.nside2npix(self.nside)
        self.z_edges = config['z_edges']

        self.cat_data = self._bin_z(self.cat_data)
        self.cat_random = self._bin_z(self.cat_random)
        self.w_data = self._get_weights(self.cat_data)
        self.w_random = self._get_weights(self.cat_random)
        self.alpha = np.sum(self.w_data)/np.sum(self.w_random)

        self.dndz       = None
        self.delta_map  = None
        self.nl_coupled = None
        self.nmt_field  = None
        self.mask = self._fill_mask()

    def _bin_z(self, cat):
        #Account for randoms using different nomenclature
        if 'ZREDMAGIC' in cat:
            z_key= 'ZREDMAGIC'
        else:
            z_key = 'Z'
            
        return cat[(cat[z_key] >= self.z_edges[0]) &
                   (cat[z_key] < self.z_edges[1])]

    
    def _get_weights(self, cat):
        #Account for randoms having no weights 
        if 'weight' in cat:
            weights = np.array(cat['weight'].values)
        else:
            weights = np.ones(len(cat))
        return weights

    def _get_counts_map(self, cat, w, nside=None):
        if nside is None:
            nside = self.nside
        npix = hp.nside2npix(nside)    
        ipix = hp.ang2pix(nside, cat['RA'], cat['DEC'],
                          lonlat=True)
        numcount = np.bincount(ipix, w, npix)
        
        return numcount
    
    def get_mask(self):
        return self.mask
    
    def _fill_mask(self):
        self.filled_mask = np.zeros(self.npix)
        goodpix = self.mask > 0
        self.filled_mask[goodpix] = self.mask[goodpix]
        return self.filled_mask
        
    def get_nz(self, num_z=200):
        if self.dndz is None:
            h, b = np.histogram(self.cat_data['ZREDMAGIC'], bins=num_z,
                                weights=self.w_data)
            self.dndz = np.array([h, b[:-1], b[1:]])
        return self.dndz

    def get_signal_map(self):
        if self.delta_map is None:
            self.delta_map = np.zeros(self.npix)
            nmap_data = self._get_counts_map(self.cat_data, self.w_data)
            nmap_random = self._get_counts_map(self.cat_random, self.w_random)
            goodpix = self.mask > 0
            self.delta_map[goodpix] = (nmap_data[goodpix] - self.alpha * nmap_random[goodpix])/self.mask[goodpix]
        return self.delta_map

    def get_nmt_field(self, signal, mask):
        if self.nmt_field is None:
            self.nmt_field = nmt.NmtField(mask, [signal], n_iter = 0)
        return self.nmt_field

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            pixel_A =  4*np.pi/hp.nside2npix(self.nside)
            N_ell = pixel_A**2*(np.sum(self.w_data**2)+ self.alpha**2*np.sum(self.w_random**2))/(4*np.pi)
            self.nl_coupled = np.array([N_ell * np.ones(3*self.nside)])
        return self.nl_coupled
