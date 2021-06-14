import glob
import pytest
import os
import yaml
from xcell.cls.data import Data


def read_yaml_file(file):
    with open(file) as f:
        config = yaml.safe_load(f)
    return config


def get_input_file():
    return './xcell/tests/data/desy1_ebossqso_p18cmbk.yml'


def get_config_dict():
    input_file = get_input_file()
    config = read_yaml_file(input_file)
    return config


def get_data():
    input_file = get_input_file()
    return Data(input_file)


def get_data_from_dict():
    input_dict = get_config_dict()
    return Data(data=input_dict)


def remove_yml_file(config):
    outdir = config['output']
    fname = os.path.join(outdir, '*.yml')
    for f in glob.glob(fname):
        os.remove(f)


def test_initizalization():
    input_file = get_input_file()
    config = read_yaml_file(input_file)
    with pytest.raises(ValueError):
        Data(input_file, data=config)

    config['bpw_edges'] = [0, 10]
    data2 = Data(data=config, override=True)
    assert os.path.isfile('xcell/tests/cls/data.yml')
    assert data2.data['bpw_edges'] == [0, 10]
    remove_yml_file(config)


def test_read_data():
    data = get_data()
    input_file = get_input_file()
    config = get_config_dict()
    assert data.read_data(input_file) == config

    remove_yml_file(config)


def test_read_saved_data():
    data = get_data()
    data2 = get_data()
    assert data.data == data2.data
    remove_yml_file(data.data)


def test_get_tracers_used():
    data = get_data()
    config = data.data
    tracers_for_cl = data.get_tracers_used()

    # In the xcell/tests/data/desy1_ebossqso_p18cmbk.yml file no eBOSS tracer
    # is used.
    trs = ['DESgc', 'DESwl', 'PLAcv']
    tracers_for_cl_test = []
    for tr in config['tracers']:
        if tr.split('__')[0] in trs:
            tracers_for_cl_test.append(tr)

    assert tracers_for_cl == tracers_for_cl_test
    assert data.get_tracers_used(True) == ['DESgc__0', 'DESwl__0', 'DESwl__1',
                                           'DESwl__2', 'DESwl__3', 'PLAcv']
    remove_yml_file(data.data)


def test_get_tracer_bare_name():
    data = get_data()
    config = data.data

    for tr in config['tracers']:
        bname = data.get_tracer_bare_name(tr)
        assert tr.split('__')[0] == bname
    remove_yml_file(data.data)


def test_get_cl_trs_names():
    data = get_data()
    config = data.data

    def assert_cls(wsp=False):
        cl_trs = data.get_cl_trs_names(wsp)

        cl_trs_test = []
        trs = data.get_tracers_used(wsp)
        for i, tri in enumerate(trs):
            for trj in trs[i:]:
                bn1 = data.get_tracer_bare_name(tri)
                bn2 = data.get_tracer_bare_name(trj)
                key = '-'.join([bn1, bn2])
                if (config['cls'][key]['compute'] == 'auto') and (tri == trj):
                    cl_trs_test.append((tri, trj))
                elif config['cls'][key]['compute'] == 'all':
                    cl_trs_test.append((tri, trj))

        assert cl_trs == cl_trs_test

    assert_cls()
    assert_cls(True)
