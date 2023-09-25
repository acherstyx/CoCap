# -*- coding: utf-8 -*-
# @Time    : 2023/2/19 23:23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : registry.py


from fvcore.common.registry import Registry
from typing import Any


class LooseRegistry(Registry):
    def get(self, name: str) -> Any:
        ret = None
        for obj_name, obj in self._obj_map.items():
            if name in obj_name:
                ret = obj
                break
        if ret is None:
            raise KeyError(
                "No object contains '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret


class PrefixRegistry(Registry):
    def get(self, name: str) -> Any:
        ret = None
        for obj_name, obj in self._obj_map.items():
            if obj_name.startswith(name):
                ret = obj
                break
        if ret is None:
            raise KeyError(
                "No object starts with '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret


class PostfixRegistry(Registry):
    def get(self, name: str) -> Any:
        ret = None
        for obj_name, obj in self._obj_map.items():
            if obj_name.endswith(name):
                ret = obj
                break
        if ret is None:
            raise KeyError(
                "No object ends with '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret
