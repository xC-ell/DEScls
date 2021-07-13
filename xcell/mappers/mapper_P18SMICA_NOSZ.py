from .mapper_base import MapperBase
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits
import os


class MapperP18SMICA_NOSZ(MapperBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map': [path+'COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits'], 
         '', 
         'nside':512}
        """
        self._get_defaults(config)
        self.file_map = config['file_map']
        self.file_hm1 = config.get('file_hm1', None)
        self.file_hm2 = config.get('file_hm2', None)
        self.file_mask = config['file_mask']
        self.signal_map = None
        self.hm1_map = None
        self.hm2_map = None
        self.diff_map = None
        self.nl_coupled = None
        self.mask = None 

    def _get_bands(self):
        while i <= 3 * self.nside:
            ells.append(round(i))
            #i = i*(1+i/(3 * nside))
            i = i+20*(1+i/240)

        self.bands = nmt.NmtBin.from_edges(ells[:-1], ells[1:])
        self.ell_arr = bands.get_effective_ells()
        return 
        
    def get_signal_map(self):
        if self.signal_map is None:
            signal_map =  hp.read_map(self.file_map)
            self.signal_map = [hp.ud_grade(signal_map,
                                           nside_out=self.nside)]
        return self.signal_map
    
    def _get_hm_maps(self):
        if self.hm1_map is None:
            hm1_map =  hp.read_map(self.file_hm1)
            self.hm1_map = [hp.ud_grade(hm1_map,
                                        nside_out=self.nside)]
        if self.hm2_map is None:
            hm2_map =  hp.read_map(self.file_hm2)
            self.hm2_map = [hp.ud_grade(hm2_map,
                                        nside_out=self.nside)]
        return self.hm1_map, self.hm2_map

    def _get_diff_map(self):
        if self.diff_map is None:
            self.hm1_map, self.hm2_map = self._get_hm_maps()
            self.diff_map = self.hm1_map[0] - self.hm2_map[0]
        return [self.diff_map]

    def get_mask(self):
        if self.mask is None:
            self.mask = np.ones(12*self.nside**2)
            for key in self.file_mask.keys():
                file = self.file_mask[key]
                mask = hp.read_map(file)
                mask = hp.ud_grade(mask,
                                   nside_out=self.nside)  
               #if key=='points':
               #     mask = hp.read_map(file)
               #     mask = hp.ud_grade(mask,
               #                    nside_out=self.nside)
               #elif key=='galactic':
                    # This is how it should be done but it returns crap
                    # masks = fits.open(file)[1].datafile
                    # mask = masks['GAL060']
               #     mask = hp.read_map(file)
               #     mask = hp.ud_grade(mask,
               #                        nside_out=self.nside)  
               
                # No quite right but cannot figure it out now
                self.mask[self.mask>0] = (1-1*abs(self.mask-mask))[self.mask>0]
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.diff_map = self._get_diff_map()
            diff_f = self.get_nmt_field(signal=self.diff_map)
            self.nl_coupled = nmt.compute_coupled_cell(diff_f, diff_f)/4
        return self.nl_coupled

    def get_beam(self):
        return self.beam

    def get_dtype(self):
        return 'cmb_SMICA_NOSZ'

    def get_spin(self):
        return 0
