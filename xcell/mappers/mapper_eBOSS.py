from .mapper_SDSS import MapperSDSS


class MappereBOSS(MapperSDSS):
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
        self._get_SDSS_defaults(config)
        self.z_edges = config.get('z_edges', [0, 3])
        self.w_method = 'eBOSS'
