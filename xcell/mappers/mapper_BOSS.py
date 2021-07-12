from .mapper_SDSS import MapperSDSS


class MapperBOSS(MapperSDSS):
    def __init__(self, config):
        """
        config - dict
        {'data_catalogs':[data_path+'BOSSLOWZ/galaxy_DR12v5_LOWZ_North.fits'],
        'random_catalogs':[data_path+'BOSSLOWZ/random1_DR12v5_LOWZ_North.fits'],
        'mask_path':[data_path+'BOSSLOWZ/mask_DR12v5_LOWZ_North.fits'],
        'z_edges':[0.15, 0.43] / [0.43, 0.75]
        'nside':nside, 'nside_mask': nside_mask, 'mask_name': 'mask_LOWZ_NGC'}
        """
        self._get_SDSS_defaults(config)
        self.z_edges = config.get('z_edges', [0, 1])
        self.w_method = 'BOSS'
