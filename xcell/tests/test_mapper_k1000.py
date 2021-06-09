import numpy as np
import xcell as xc
import healpy as hp
import os


def get_config(w_stars=False):
    if w_stars:
        fname = 'xcell/tests/data/catalog_stars.fits'
    else:
        fname = 'xcell/tests/data/catalog.fits'
    return {'data_catalog': fname,
            'file_nz': 'xcell/tests/data/Nz_DIR_z0.1t0.3.asc',
            'zbin': 0, 'nside': 32, 'mask_name': 'mask',
            'e1_flag': 'bias_corrected_e1',
            'e2_flag': 'bias_corrected_e2'}


def get_mapper(w_stars=False):
    return xc.mappers.MapperKiDS1000(get_config(w_stars))


def test_smoke():
    get_mapper()


def get_es():
    npix = hp.nside2npix(32)
    return np.repeat(np.array([np.arange(4)]), npix//4,
                     axis=0).flatten()


def test_lite():
    predir = 'xcell/tests/data/'
    config = get_config()
    config['path_lite'] = predir
    ifile = 0
    while os.path.isfile(f'{predir}KiDS1000_lite_cat_zbin{ifile}.fits'):
        os.remove(f'{predir}KiDS1000_lite_cat_zbin{ifile}.fits')
        ifile += 1
    m = xc.mappers.MapperKiDS1000(config)
    m.get_catalog()
    assert os.path.isfile(f'{predir}KiDS1000_lite_cat_zbin0.fits')

    # Non-exsisting fits files - read from lite
    config['data_catalog'] = 'whatever'
    xc.mappers.MapperKiDS1000(config)


def test_get_signal_map():
    m = get_mapper()
    ms = get_mapper(w_stars=True)
    sh = np.array(m.get_signal_map('shear'))
    psf = np.array(m.get_signal_map('PSF'))
    star = np.array(ms.get_signal_map('stars'))
    es = get_es()
    assert sh.shape == (2, hp.nside2npix(32))
    assert psf.shape == (2, hp.nside2npix(32))
    assert star.shape == (2, hp.nside2npix(32))
    assert np.all(np.fabs(-sh+(np.mean(es)-es)/(1+m.m[0])) < 1E-5)
    assert np.all(np.fabs(-psf-es) < 1E-5)
    assert np.all(np.fabs(-star-es) < 1E-5)


def test_get_mask():
    m = get_mapper()
    ms = get_mapper(w_stars=True)
    sh = m.get_mask('shear')
    psf = m.get_mask('PSF')
    star = ms.get_mask('stars')
    assert len(sh) == len(psf) == len(star) == hp.nside2npix(32)
    assert np.all(np.fabs(sh-2) < 1E-5)
    assert np.all(np.fabs(psf-2) < 1E-5)
    assert np.all(np.fabs(star-2) < 1E-5)


def test_get_nl_coupled():
    m = get_mapper()
    ms = get_mapper(w_stars=True)
    aa = hp.nside2pixarea(32)

    sh = m.get_nl_coupled()
    shp = 4*np.std(np.arange(4))**2*aa/(1+m.m[0])**2
    assert np.all(sh[0][:2] == 0)
    assert np.fabs(np.mean(sh[0][2:])-shp) < 1E-5

    psf = m.get_nl_coupled('PSF')
    psfp = 4*np.mean(np.arange(4)**2)*aa
    assert np.all(psf[0][:2] == 0)
    assert np.fabs(np.mean(psf[0][2:])-psfp) < 1E-5

    star = ms.get_nl_coupled('stars')
    starp = 4*np.mean(np.arange(4)**2)*aa
    assert np.all(star[0][:2] == 0)
    assert np.fabs(np.mean(star[0][2:])-starp) < 1E-5
