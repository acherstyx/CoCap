# -*- coding: utf-8 -*-
# @Time    : 2022/12/1 03:01
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : show_registry.py


from fvcore.common.registry import Registry
from tabulate import tabulate


def reformat_registry_output(self: Registry):
    table_headers = ["Names", "Objects"]
    table = tabulate(
        self._obj_map.items(), headers=table_headers, tablefmt="psql"
    )
    return "Registry of {}:\n".format(self._name) + table


def main():
    from mm_video.config.base import CUSTOM_CONFIG_REGISTRY, CUSTOM_CONFIG_CHECK_REGISTRY
    from mm_video.data.build import DATASET_REGISTRY, COLLATE_FN_REGISTER
    from mm_video.modeling.model import MODEL_REGISTRY
    from mm_video.modeling.optimizer import OPTIMIZER_REGISTRY
    from mm_video.modeling.loss import LOSS_REGISTRY
    from mm_video.modeling.meter import METER_REGISTRY

    for reg in (CUSTOM_CONFIG_REGISTRY, CUSTOM_CONFIG_CHECK_REGISTRY, DATASET_REGISTRY, COLLATE_FN_REGISTER,
                MODEL_REGISTRY, OPTIMIZER_REGISTRY, LOSS_REGISTRY, METER_REGISTRY):
        print(reformat_registry_output(reg))


if __name__ == '__main__':
    main()
