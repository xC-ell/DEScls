import os
import pytest
import numpy as np
import sacc
import shutil
from xcell.cls.to_sacc import sfile
from xcell.cls.data import Data
from xcell.cls.cl import Cl, ClFid
from xcell.cls.cov import Cov

# Remove previous test results
tmpdir = './xcell/tests/cls/dummy1'


def remove_tmpdir():
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)


def get_config(fsky0=0.2, fsky1=0.3, dtype0='galaxy_density',
               dtype1='galaxy_shear'):
    nside = 32
    # Set only the necessary entries. Leave the others to their default
    # value.
    cosmo = {
            # Planck 2018: Table 2 of 1807.06209
            # Omega_m: 0.3133
            'Omega_c': 0.2640,
            'Omega_b': 0.0493,
            'h': 0.6736,
            'n_s': 0.9649,
            'sigma8': 0.8111,
            'w0': -1,
            'wa': 0,
            'transfer_function': 'eisenstein_hu',
            'baryons_power_spectrum': 'nobaryons',
        }
    dummy0 = {'mask_name': 'mask_dummy0', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky0, 'seed': 0,
              'dtype': dtype0}
    dummy1 = {'mask_name': 'mask_dummy1', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky1, 'seed': 100,
              'dtype': dtype1}
    bpw_edges = list(range(0, 3 * nside, 4))

    return {'tracers': {'Dummy__0': dummy0, 'Dummy__1': dummy1},
            'cls': {'Dummy-Dummy': {'compute': 'all'}},
            'cov': {'fiducial': {'cosmo': cosmo, 'gc_bias':  False, 'wl_m':
                                 False, 'wl_ia': False}},
            'bpw_edges': bpw_edges,
            'healpy': {'n_iter_sht': 0, 'n_iter_mcm': 3, 'n_iter_cmcm': 3,
                       'nside': nside},
            'recompute': {'cls': False, 'cov': False, 'mcm': False, 'cmcm':
                          False},
            'output': tmpdir}


def get_data(fsky=0.2, fsky2=0.3, dtype0='galaxy_density',
             dtype1='galaxy_shear'):
    config = get_config(fsky, fsky2, dtype0, dtype1)
    return Data(data=config)


def get_sfile(use='cls', m_marg=False, fsky0=0.2, fsky1=0.3, dtype0='galaxy_density',
               dtype1='galaxy_shear'):
    # Generate data.yml file
    data = get_data(dtype0=dtype0, dtype1=dtype1)
    datafile = os.path.join(data.data['output'], 'data.yml')
    return sfile(datafile, 'cls_cov_dummy.fits', use, m_marg)


@pytest.mark.parametrize('use', ['cls', 'nl', 'fiducial', 'potato'])
def test_init(use):
    outdir = get_config()['output']
    if use != 'potato':
        get_sfile(use)
        sfile_path = os.path.join(outdir, 'cls_cov_dummy.fits')
        assert os.path.isfile(sfile_path)
    else:
        with pytest.raises(ValueError):
            get_sfile(use)



@pytest.mark.parametrize('dt1, dt2', [('galaxy_density', 'galaxy_shear'),
                                      ('galaxy_shear', 'cmb_convergence'),
                                      ('galaxy_density', 'cmb_convergence')])
def test_added_tracers(dt1, dt2):
    s = get_sfile(dtype0=dt1, dtype1=dt2)
    data = get_data(dtype0=dt1, dtype1=dt2)
    for trname in data.data['tracers'].keys():
        tr = s.s.tracers[trname]
        m = data.get_mapper(trname)
        assert tr.quantity == m.get_dtype()
        if tr.quantity in ['galaxy_density', 'galaxy_shear']:
            assert isinstance(tr, sacc.tracers.NZTracer)
            z, nz = m.get_nz()
            assert np.all(tr.z == z)
            assert np.all(tr.nz == nz)
        elif tr.quantity in ['cmb_convergence']:
            assert isinstance(tr, sacc.tracers.MapTracer)
            assert tr.ell == m.get_ell()
            # Only here because tr.spin is not an attribute of NZTracers
            assert tr.spin == m.get_spin()
            assert tr.beam_extra['nl'] == m.get_nl_coupled()[0]
            assert tr.beam == np.ones_like(tr.ell)
        else:
            raise ValueError('Tracer not implemented')


@pytest.mark.parametrize('use', ['cls', 'nl', 'fiducial'])
def test_ell_cl_autocov(use):
    s = get_sfile(use)
    data = get_data()


    for dtype in s.s.get_data_types():
        for trs in s.s.get_tracer_combinations(dtype):
            ixd = {'cl_00': 0, 'cl_0e': 0, 'cl_0b': 1, 'cl_ee': 0}
            if dtype in ['cl_00', 'cl_0e', 'cl_ee']:
                ix = 0
            elif dtype in ['cl_0b', 'cl_eb']:
                ix = 1
            elif dtype in ['cl_be']:
                ix = 2
            elif dtype in ['cl_bb']:
                ix = 3
            else:
                raise ValueError(f'data_type {dtype} must be weird!')

            if use == 'fiducial':
                cl_class = ClFid(data.data, *trs)
            else:
                cl_class = Cl(data.data, *trs)

            if use == 'nl':
                ell, cl = s.s.get_ell_cl(dtype, trs[0], trs[1])
                ell2, cl2 = cl_class.get_ell_nl()
            else:
                ell, cl, cov = s.s.get_ell_cl(dtype, trs[0], trs[1],
                                              return_cov=True)
                ell2, cl2 = cl_class.get_ell_cl()

                cov_class = Cov(data.data, *trs, *trs)
                nbpw = ell.size
                ncls = cl2.shape[0]
                cov2 = cov_class.get_covariance()
                cov2 = cov2.reshape((nbpw, ncls, nbpw, ncls))[:, ix, :, ix]
                assert np.max(np.abs((cov - cov2) / np.mean(cov))) < 1e-5


            if use == 'fiducial':
                # Matrices to bin the fiducial Cell
                ws_bpw = np.zeros((ell.size, ell2.size))
                ws_bpw[np.arange(ell.size), ell.astype(int)] = 1
                assert np.all(cl == ws_bpw.dot(cl2[ix]))
            else:
                assert np.all(cl == cl2[ix])
                assert np.all(ell == ell2)


def test_get_dof_tracers():
    s = get_sfile()
    for tr1, tr2 in s.s.get_tracer_combinations():
        s1 = np.max((s.data.get_mapper(tr1).get_spin(), 1))
        s2 = np.max((s.data.get_mapper(tr2).get_spin(), 1))

        dof = s1 * s2
        assert dof == s.get_dof_tracers((tr1, tr2))


def test_get_datatypes_from_dof():
    s = get_sfile()
    assert s.get_datatypes_from_dof(1) == ['cl_00']
    assert s.get_datatypes_from_dof(2) == ['cl_0e', 'cl_0b']
    assert s.get_datatypes_from_dof(4) == ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
    with pytest.raises(ValueError):
        s.get_datatypes_from_dof(3)


if os.path.isdir(tmpdir):
    shutil.rmtree(tmpdir)
