#!/usr/bin/python
from cl import Cl
from cov import Cov
from common import Data
from mappers import mapper_from_name
import numpy as np
import sacc
import os
import warnings

# TODO: move this to data.ylm?

class sfile():
    def __init__(self, datafile, output, use_nl=False):
        self.data = Data(data_path=datafile)
        self.outdir = self.data.data['output']
        self.use_nl = use_nl
        self.s = sacc.Sacc()
        self.add_tracers()
        self.add_ell_cls()
        self.add_covariance()
        fname = os.path.join(self.outdir, output)
        self.s.save_fits(fname, overwrite=True)

    def add_tracers(self):
        tracers = self.data.get_tracers_used()
        for tr in tracers:
            self.add_tracer(tr)

    def add_ell_cls(self):
        cl_tracers = self.data.get_cl_trs_names()
        for tr1, tr2 in cl_tracers:
            self.add_ell_cl(tr1, tr2)

    def read_covariance_NG(self):
        dtype = self.s.get_data_types()[0]
        cl_tracers = self.s.get_tracer_combinations(data_type=dtype)
        ell, _ = self.s.get_ell_cl(dtype, *cl_tracers[0])
        nbpw = ell.size
        #
        cl_ng_tracers = self.data.get_cov_ng_cl_tracers()
        ncls = len(cl_ng_tracers)
        #
        cov_ng = self.data.data['cov']['ng']
        cov = np.load(cov_ng['path']).reshape((ncls, nbpw, ncls, nbpw))

        ndim = self.s.mean.size
        covmat = np.zeros((int(ndim/nbpw), nbpw, int(ndim/nbpw), nbpw))
        print(ndim/nbpw)
        cl_tracers = self.s.get_tracer_combinations()
        for i, trs1 in enumerate(cl_tracers):
            ix1 = cl_ng_tracers.index(trs1)
            cl_ix1 = int(self.s.indices(tracers=trs1)[0] / nbpw)
            for j, trs2 in enumerate(cl_tracers[i:], i):
                ix2 = cl_ng_tracers.index(trs2)
                cl_ix2 = int(self.s.indices(tracers=trs2)[0] / nbpw)
                covi = cov[ix1, :, ix2, :]
                covmat[cl_ix1, :, cl_ix2, :] = covi
                covmat[cl_ix2, :, cl_ix1, :] = covi.T
        print(self.s.indices(tracers=trs1))
        return covmat.reshape((ndim, ndim))

    def add_covariance_NG(self):
        covmat = self.read_covariance_NG()
        self.s.add_covariance(covmat)

    def add_covariance_G(self):
        # Get nbpw
        dtype = self.s.get_data_types()[0]
        tracers = self.s.get_tracer_combinations(data_type=dtype)[0]
        ell, _ = self.s.get_ell_cl(dtype, *tracers)
        nbpw = ell.size
        #
        ndim = self.s.mean.size
        cl_tracers = self.s.get_tracer_combinations()

        covmat = -1 * np.ones((ndim, ndim))
        # def dic(tr):
        #     ibin = int(tr[-1])
        #     if 'gc' in tr:
        #         s = 0
        #     elif 'wl' in tr:
        #         s = 2
        #         # ibin += 5
        #     elif 'cv' in tr:
        #         s = 0
        #         ibin = 9
        #     return ibin, s

        for i, trs1 in enumerate(cl_tracers):
            dof1 = self.get_dof_tracers(trs1)
            dtypes1 = self.get_datatypes_from_dof(dof1)
            for trs2 in cl_tracers[i:]:
                dof2 = self.get_dof_tracers(trs2)
                dtypes2 = self.get_datatypes_from_dof(dof2)
                print(trs1, trs2)

                # tr1, tr2 = trs1
                # tr3, tr4 = trs2
                # b1, s1 = dic(tr1)
                # b2, s2 = dic(tr2)
                # b3, s3 = dic(tr3)
                # b4, s4 = dic(tr4)

                cov = Cov(self.data.data, *trs1, *trs2).get_covariance().reshape((nbpw, dof1, nbpw, dof2))
                # cname = f'cov_s{s1}{s2}{s3}{s4}_b{b1}{b2}{b3}{b4}.npz'
                # cov = np.load('/mnt/extraspace/gravityls_3/S8z/Cls/all_together/new_fiducial_cov/' + cname)['arr_0'].reshape((nbpw, dof1, nbpw, dof2))
                # cov = np.load(f'/mnt/extraspace/damonge/S8z_data/outputs/cls_metacal_covar_bins_new_nka_full_noise_{b1}{b2}_{b3}{b4}_ns4096.npz')['cov']

                for i, dt1 in enumerate(dtypes1):
                    ix1 = self.s.indices(tracers=trs1, data_type=dt1)
                    if len(ix1) == 0:
                        continue
                    for j, dt2 in enumerate(dtypes2):
                        ix2 = self.s.indices(tracers=trs2, data_type=dt2)
                        if len(ix2) == 0:
                            continue
                        covi = cov[:, i, :, j]
                        covmat[np.ix_(ix1, ix2)] = covi
                        covmat[np.ix_(ix2, ix1)] = covi.T

        # covmat += self.read_covariance_NG()
        self.s.add_covariance(covmat)

    def add_covariance(self):
        if self.use_nl:
            self.add_covariance_NG()
        else:
            self.add_covariance_G()

    def add_tracer(self, tr):
        config = self.data.data['tracers'][tr]
        mapper_class = config['mapper_class']
        mapper = mapper_from_name(mapper_class)(config)
        quantity = mapper.get_dtype()
        spin = mapper.get_spin()
        if quantity == 'galaxy_density':
            z, nz = mapper.get_nz(dz=0)
            self.s.add_tracer('NZ', tr, quantity=quantity, spin=spin,
                              z=z, nz=nz)
        elif quantity == 'galaxy_shear':
            z, nz = mapper.get_nz(dz=0)
            self.s.add_tracer('NZ', tr, quantity=quantity, spin=spin,
                              z=z, nz=nz)
        elif quantity == 'cmb_convergence':
            # TODO: Check
            # ell, nl = np.loadtxt(tracer['nl'], usecols=tracer['nl_cols'], unpack=True)
            ell = mapper.get_ells()
            nl = mapper.get_nl_coupled()

            beam = np.ones_like(ell)
            self.s.add_tracer('Map', tr, quantity=quantity, spin=spin,
                              ell=ell, beam=beam, beam_extra={'nl': nl})
        else:
            raise ValueError('Tracer type {} not implemented'.format(quantity))


    def add_ell_cl(self, tr1, tr2):
        ells_nobin = np.arange(3 * self.data.data['healpy']['nside'])
        cl = Cl(self.data.data, tr1, tr2)
        cl.get_cl_file()
        if not self.use_nl:
            w = cl.get_workspace()
            ws_bpw = w.get_bandpower_windows()

        cl_types = self.get_datatypes_from_dof(cl.cl.shape[0])

        for i, cl_type in enumerate(cl_types):
            if (cl_type == 'cl_be') and (tr1 == tr2):
                continue
            elif self.use_nl:
                cli = cl.nl[i]
                wins = None
            else:
                # b1 = int(tr1[-1])
                # b2 = int(tr2[-1])
                # predir = '/mnt/extraspace/damonge/S8z_data/outputs/'
                # fname_win = predir + f'cls_metacal_win_bins_{b1}{b2}_ns4096.npz'
                # ws_bpw = np.load(fname_win)['win']
                wins = sacc.BandpowerWindow(ells_nobin, ws_bpw[i, :, i, :].T)
                cli = cl.cl[i]
                # fname_cl = predir + f'cls_metacal_cls_bins_{b1}{b2}_ns4096.npz'
                # cload = np.load(fname_cl)
                # cli = cload['cls'][i] - cload['nls'][i]
                # ls = cload['ls']

            self.s.add_ell_cl(cl_type, tr1, tr2, cl.ell, cli, window=wins)
            # self.s.add_ell_cl(cl_type, tr1, tr2, ls, cli, window=wins)

    def get_dof_tracers(self, tracers):
        tr1, tr2 = tracers
        s1, s2 = Cl(self.data.data, tr1, tr2).get_spins()

        dof = s1 + s2
        if dof == 0:
            dof += 1

        return dof

    def get_datatypes_from_dof(self, dof):
        if dof == 1:
            cl_types = ['cl_00']
        elif dof == 2:
            cl_types = ['cl_0e', 'cl_0b']
        elif dof == 4:
            cl_types = ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
        else:
            raise ValueError('dof does not match 1, 2, or 4.')

        return cl_types

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('name', type=str, help="Name of the generated sacc file. Stored in yml['output']")
    parser.add_argument('--use_nl', action='store_true', default=False, help="Set if you want to use nl and covNG (if present) instead of cls and covG")
    args = parser.parse_args()

    sfile = sfile(args.INPUT, args.name, args.use_nl)
