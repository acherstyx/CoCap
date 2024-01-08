# -*- coding: utf-8 -*-
# @Time    : 10/11/23
# @Author  : Yaojie Shen
# @Project : MM-Video
# @File    : registry.py

from hydra_zen import ZenStore

__all__ = [
    "dataset_store", "model_store", "meter_store", "trainer_store", "runner_store",
    "DATASET_CONFIG_GROUP", "MODEL_CONFIG_GROUP", "METER_CONFIG_GROUP", "TRAINER_CONFIG_GROUP", "RUNNER_CONFIG_GROUP"
]

# a list of hard-coded registry positions for config store
DATASET_CONFIG_GROUP = "dataset"
MODEL_CONFIG_GROUP = "model"
METER_CONFIG_GROUP = "meter"
TRAINER_CONFIG_GROUP = "trainer"
RUNNER_CONFIG_GROUP = "runner"

dataset_store = ZenStore(name="dataset", deferred_hydra_store=False)(group=DATASET_CONFIG_GROUP)
model_store = ZenStore(name="model", deferred_hydra_store=False)(group=MODEL_CONFIG_GROUP)
meter_store = ZenStore(name="meter", deferred_hydra_store=False)(group=METER_CONFIG_GROUP)
trainer_store = ZenStore(name="trainer", deferred_hydra_store=False)(group=TRAINER_CONFIG_GROUP)
runner_store = ZenStore(name="runner", deferred_hydra_store=False)(group=RUNNER_CONFIG_GROUP)
