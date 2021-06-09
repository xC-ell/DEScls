# flake8: noqa
from .mapper_base import MapperBase
from .mapper_DESY1gc import MapperDESY1gc
from .mapper_DESY1wl import MapperDESY1wl
from .mapper_eBOSSQSO import MappereBOSSQSO
from .mapper_KV450 import MapperKV450
from .mapper_KiDS1000 import MapperKiDS1000
from .mapper_P18CMBK import MapperP18CMBK
from .mapper_DECaLS import MapperDECaLS
from .mapper_2MPZ import Mapper2MPZ
from .mapper_WIxSC import MapperWIxSC
from .mapper_dummy import MapperDummy
from .utils import get_map_from_points, get_DIR_Nz


def mapper_from_name(name):
    mappers = {m.__name__: m for m in MapperBase.__subclasses__()}
    if name in mappers:
        return mappers[name]
    else:
        raise ValueError(f"Unknown mapper {name}")
