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

    def get_contaminants(self):
        return None

    def get_mask(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_coupled(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_covariance(self):
        raise NotImplementedError("Do not use base class")

    def get_nmt_field(self, signal=None, **kwargs):
        if self.nmt_field is None:
            if signal is None:
                signal = self.get_signal_map(**kwargs)
            mask = self.get_mask(**kwargs)
            cont = self.get_contaminants(**kwargs)
            n_iter = kwargs.get('n_iter', 0)
            self.nmt_field = nmt.NmtField(mask, signal,
                                          templates=cont, n_iter=n_iter)
        return self.nmt_field
