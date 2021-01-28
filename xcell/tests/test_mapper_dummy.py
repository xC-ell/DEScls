import numpy as np
import xcell as xc
import healpy as hp


def get_config(spin=0):
    return {'l0': 100,
            'alpha': -2,
            'spin': spin}


def get_mapper():
    return xc.mappers.MapperDummy(get_config())

def test_smoke():
    get_mapper()


def test_get_mask():
    m = get_mapper()
    d = m.get_mask(fsky=1)
    assert np.all(d == 1)
    d = m.get_mask(fsky=100)
    assert np.all(d == 1)
    # TODO: Implement apodized mask
    # d = m.get_mask(fsky=0.5)

def test_get_cl():
    m = get_mapper()
    c = get_config()
    ls = 120
    cl = 1./(ls+c['l0'])**c['alpha']
    assert(cl == m.get_cl(ls))

def test_get_signal_map():
    m = get_mapper()
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert np.all(np.fabs(d) < 1E-5)


def test_get_nl_coupled():
    m = get_mapper()
    # Redicted value
    nl_pred = 4*np.pi/m.npix
    nl = m.get_nl_coupled()
    assert np.all(np.fabs(nl-nl_pred) < 1E-5)


def test_get_dtype():
    m = get_mapper()
    assert 'generic' == m.get_dtype()


def test_get_spin():
    m = get_mapper()
    c = get_config()
    assert c['spin'] == m.get_spin()
