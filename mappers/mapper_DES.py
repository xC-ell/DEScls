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

        self.cat_data = Table.read(self.config['data_catalogs']).to_pandas()  
        self.mask = hp.read_map(self.config['file_mask'], verbose = False)  
        self.nz = Table.read(self.config['file_nz']).to_pandas()
            
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.z_edges = config['z_edges']

        self.cat_data = self._bin_z(self.cat_data)
        self.w_data = self._get_weights(self.cat_data)
        self.nmap_data = self._get_counts_map(self.cat_data, self.w_data)  

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
    
    def _fill_mask(self):
        self.filled_mask = np.zeros(self.npix)
        goodpix = self.mask > 0
        self.filled_mask[goodpix] = self.mask[goodpix]
        return self.filled_mask
    
    def get_mask(self):
        goodpix = self.mask > 0.5
        self.mask[~goodpix] = 0
        return self.mask
        
    def get_nz(self, num_z=200):
        if self.dndz is None:
            h, b = np.histogram(self.cat_data['ZREDMAGIC'], bins=num_z,
                                weights=self.w_data)
            self.dndz = np.array([h, b[:-1], b[1:]])
        return self.dndz

    def get_signal_map(self):
        if self.delta_map is None:
            self.delta_map = np.zeros(self.npix)
            N_mean = np.sum(self.nmap_data)/np.sum(self.mask)
            goodpix = self.mask > 0
            self.delta_map[goodpix] = (self.nmap_data[goodpix])/(self.mask[goodpix]*N_mean) -1
        return [self.delta_map]

    def get_nmt_field(self):
        if self.nmt_field is None:
            signal = self.get_signal_map()
            mask = self.get_mask()
            self.nmt_field = nmt.NmtField(mask, signal, n_iter = 0)
        return self.nmt_field

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            N_mean = np.sum(self.nmap_data)/np.sum(self.mask)
            N_mean_srad = N_mean / (4 * np.pi) * self.npix
            correction = np.sum(self.w_data**2) / np.sum(self.w_data)
            N_ell = correction * np.sum(self.mask) / self.npix / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled
