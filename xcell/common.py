#!/usr/bin/python
import yaml

class Data():
    def __init__(self, data_path='', data={}):
        if (data_path) and (data):
            raise ValueError('Only one of data_path or data must be given. Both set.')
        elif data_path:
            self.data_path = data_path
            self.data = self.read_data(data_path)
        elif data:
            self.data_path = None
            self.data = data
        else:
            raise ValueError('One of data_path or data must be set. None set.')

    def read_data(self, data_path):
        with open(data_path) as f:
            data = yaml.safe_load(f)
        return data

    def get_tracers_used(self, wsp=False):
        tracers = []
        for trk, trv in self.data['cls'].items():
            tr1, tr2 = trk.split('-')
            if trv['compute'] != 'None':
                tracers.append(tr1)
                tracers.append(tr2)

        tracers_for_cl = []
        for tr in self.data['tracers'].keys():
            tr_nn = ''.join(s for s in tr if not s.isdigit())
            if tr_nn in tracers:
                tracers_for_cl.append(tr)

        if wsp:
            tracers_for_cl = self.filter_tracers_wsp(tracers_for_cl)

        return tracers_for_cl

    def get_cl_trs_names(self, wsp=False):
        cl_tracers = []
        tr_names = self.get_tracers_used(wsp)  # [trn for trn in data['tracers']]
        for i, tr1 in enumerate(tr_names):
            for tr2 in tr_names[i:]:
                trreq = ''.join(s for s in (tr1 + '-' + tr2) if not s.isdigit())
                if trreq not in self.data['cls']:
                    continue
                clreq =  self.data['cls'][trreq]['compute']
                if clreq == 'all':
                    pass
                elif (clreq == 'auto') and (tr1 != tr2):
                    continue
                elif clreq == 'None':
                    continue
                cl_tracers.append((tr1, tr2))

        return cl_tracers

    def get_cov_trs_names(self, wsp=False):
        cl_tracers = self.get_cl_trs_names(wsp)
        cov_tracers = []
        for i, trs1 in enumerate(cl_tracers):
            for trs2 in cl_tracers[i:]:
                cov_tracers.append((*trs1, *trs2))

        return cov_tracers

    def get_cov_ng_cl_tracers(self):
        cl_tracers = self.get_cl_trs_names()
        order_ng = self.data['cov']['ng']['order']
        cl_ng = [[] for i in order_ng]
        ix_reverse = []

        for tr1, tr2 in cl_tracers:
            tr1_nn = ''.join(s for s in tr1 if not s.isdigit())
            tr2_nn = ''.join(s for s in tr2 if not s.isdigit())
            if (tr1_nn + '-' + tr2_nn) in order_ng:
                ix = order_ng.index(tr1_nn + '-' + tr2_nn)
            elif (tr2_nn + '-' + tr1_nn) in order_ng:
                ix = order_ng.index(tr2_nn + '-' + tr1_nn)
                if ix not in ix_reverse:
                    ix_reverse.append(ix)
            else:
                raise ValueError('Tracers {}-{} not found in NG cov.'.format(tr1, tr2))
            cl_ng[ix].append((tr1, tr2))

        for ix in ix_reverse:
            cl_ng[ix].sort(key=lambda x: x[1])

        return [item for sublist in cl_ng for item in sublist]


    def filter_tracers_wsp(self, tracers):
        tracers_torun = []
        masks = []
        for tr in tracers:
            mtr = self.data['tracers'][tr]['mask']
            if  mtr not in masks:
                tracers_torun.append(tr)
                masks.append(mtr)

        return tracers_torun

    def get_dof_tracers(self, tracers):
        tr1, tr2 = tracers
        s1 = self.data['tracers'][tr1]['spin']
        s2 = self.data['tracers'][tr2]['spin']

        dof = s1 + s2
        if dof == 0:
            dof += 1

        return dof

    def check_toeplitz(self, dtype):
        if ('toeplitz' in self.data) and \
            (dtype in self.data['toeplitz']):
            toeplitz = self.data['toeplitz'][dtype]

            l_toeplitz = toeplitz['l_toeplitz']
            l_exact = toeplitz['l_exact']
            dl_band = toeplitz['dl_band']
        else:
            l_toeplitz = l_exact = dl_band = -1

        return l_toeplitz, l_exact, dl_band

