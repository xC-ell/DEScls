from .mapper_base import MapperBase
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits
import os


class MapperP18CIB(MapperBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-milca-ymaps_2048_R2.00.fits', 
         'nside':512}
        """
        self._get_defaults(config)
        self.file_map = config['file_map']

        self.signal_map = None 
        self.mask = None 

    def get_signal_map(self):
        if self.signal_map is None:
            signal_map =  hp.read_map(self.file_map)
            self.signal_map = [hp.ud_grade(signal_map,
                                           nside_out=self.nside)]
        return self.signal_map

    def get_mask(self):
        if self.mask is None:
            signal_map =  hp.read_map(self.file_map)
            self.signal_map = hp.ud_grade(signal_map,
                                           nside_out=self.nside)
        return self.mask

    def get_beam(self):
        return self.beam

    def get_dtype(self):
        return 'cmb_CIB'

    def get_spin(self):
        return 0
