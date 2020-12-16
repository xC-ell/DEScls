import pymaster as nmt


class MapperBase(object):
    def __init__(self, config):
        self._get_defaults(config)

    def _get_defaults(self, config):
        self.config = config
        self.mask_name = config.get('mask_name', None)
        self.nside = config['nside']
        self.nmt_field = None

    def get_signal_map(self):
        raise NotImplementedError("Do not use base class")

    def get_mask(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_coupled(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_covariance(self):
        raise NotImplementedError("Do not use base class")

    def get_nmt_field(self):
        if self.nmt_field is None:
            signal = self.get_signal_map()
            mask = self.get_mask()
            self.nmt_field = nmt.NmtField(mask, signal, n_iter=0)
        return self.nmt_field