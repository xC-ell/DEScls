import numpy as np
import xcell as xc
import healpy as hp


def get_config():
    return {'data_catalogs': ['xcell/tests/data/catalog.fits',
                              'xcell/tests/data/catalog.fits'],
            'random_catalogs': ['xcell/tests/data/catalog.fits',
                                'xcell/tests/data/catalog.fits'],
            'z_edges': [0, 1.5], 'nside': 32, 'mask_name': 'mask'}


def get_mappers(c=None):
    if c is None:
        c = get_config()
    mappers = [xc.mappers.MappereBOSS(c),
               xc.mappers.MapperBOSS(c)]
    return mappers


def test_get_dtype():
    mappers = get_mappers()
    for m in mappers:
        assert m.get_dtype() == 'galaxy_density'


def test_get_spin():
    mappers = get_mappers()
    for m in mappers:
        assert m.get_spin() == 0


def test_smoke():
    mappers = get_mappers()
    for m in mappers:
        m._get_w(mod='data')
        assert len(m.ws['data']) == 2*hp.nside2npix(32)


def test_get_nz():
    mappers = get_mappers()
    for m in mappers:
        z, nz = m.get_nz()
        assert len(z) == 50
        assert len(nz) == 50
        if m.w_method == 'eBOSS':
            LHS = 16*hp.nside2npix(32)
        if m.w_method == 'BOSS':
            LHS = (3/4)*16*hp.nside2npix(32)
        assert np.sum(nz) == LHS


def test_get_signal_map():
    mappers = get_mappers()
    for m in mappers:
        d = m.get_signal_map()
        assert len(d) == 1
        d = d[0]
        assert len(d) == hp.nside2npix(m.nside)
        assert np.all(np.fabs(d) < 1E-15)


def test_get_nl_coupled_data():
    c = get_config()
    c['nside_nl_threshold'] = 1
    c['lmin_nl_from_data'] = 10
    mappers = get_mappers(c)
    for m in mappers:
        nl = m.get_nl_coupled()
        assert np.all(nl == 0)


def test_get_mask():
    mappers = get_mappers()
    for m in mappers:
        print(m)
        d = m.get_mask()
        if m.w_method == 'eBOSS':
            LHS = 16
        if m.w_method == 'BOSS':
            LHS = 12
        assert np.all(np.fabs(d-LHS) < 1E-5)


def test_get_nl_coupled():
    mappers = get_mappers()
    for m in mappers:
        pix_area = 4*np.pi/hp.nside2npix(m.nside)
        # sum(w_d^2 + alpha^2 * w_r^2)
        nl_pred = 2*hp.nside2npix(32)*(8**2+1*8**2)
        nl_pred *= pix_area**2/(4*np.pi)
        if m.w_method == 'BOSS':
            nl_pred *= (3/4)**2
        nl = m.get_nl_coupled()
        assert np.all(np.fabs(nl-nl_pred) < 1E-5)
