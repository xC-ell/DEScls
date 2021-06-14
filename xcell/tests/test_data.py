import pytest
import os
import shutil
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
    fname = os.path.join(outdir, 'desy1_ebossqso_p18cmbk.yml')
    os.remove(fname)


def test_initizalization():
    input_file = get_input_file()
    config = read_yaml_file(input_file)
    with pytest.raises(ValueError):
        Data(input_file, data=config)


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
