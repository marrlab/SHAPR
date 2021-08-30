#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import typing
import json
import os

def auto_property(attr_storage_name):
    '''
    Automatically decorate the attribute with @property
    '''
    def get_attr(instance):
        return instance.__dict__[attr_storage_name]

    def set_attr(instance, value):
        instance.__dict__[attr_storage_name] = value
    
    return property(get_attr, set_attr)
       
class FrozenClass():
    '''
    For child classes. It prevent to add new attributes to the setting from outside
    and prevents giving mistaken arguments to the setting.
    '''
    __is_frozen = False 
    def __setattr__(self, key, value):
        if self.__is_frozen and not hasattr(self, key):
            raise AttributeError(f"'{key}' is not among the attributes of targeted object. It is an instance of a frozen class.")

        super().__setattr__(key, value)

    def set_attributes_with_keys(self, kwargs):
        for attr_name, attr_value in  kwargs.items():
            self.__setattr__(attr_name, attr_value)

    def __call__(self, kwargs):
        self.set_attributes_with_keys(self, kwargs)
        
    def _frozen(self):
        self.__is_frozen = True


        
class ShapeAEConfig(FrozenClass):
    """
    Manages the configuration of the ShapeAE
    """
    __config_param_names = dict(
        path = "__path",
        result_path = "__result_path",
        pretrained_weights_path = "__pretrained_weights_path",
        random_seed = "__random_seed",
        batch_size="__batch_size",
        epochs_ShapeAE="__epochs_ShapeAE",
        epochs_cShapeAE="__epochs_cShapeAE",
    )

    __config_param_default = dict(
        path = "",
        result_path = "",
        pretrained_weights_path = "",
        random_seed = 0,
        batch_size = 6,
        epochs_ShapeAE = 30,
        epochs_cShapeAE = 30,
    )

    def __init__(self):
        for param, storing_param in self.__config_param_names.items():
            self.__setattr__(param, auto_property(storing_param))
        self._frozen()

        # setting initial values.
        self.set_attributes_with_keys(self.__config_param_default)


    def __str__(self):
        s = "------ settings parameters ------\n"
        for _k, _ in self.__config_param_names.items():
            s += f"{_k}: {self.__getattribute__(_k)}\n"
        return s

    def read_json(self, file_path):
        with open(file_path) as f:
            _initial_values = json.load(f)
        _initial_values = {_k: _v for (_k, _v) in _initial_values.items() if not _k.startswith("_comment")}
        self.set_attributes_with_keys(_initial_values)

settings = ShapeAEConfig()

