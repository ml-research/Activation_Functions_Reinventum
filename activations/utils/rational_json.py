"""Provides functions to store and load Rationals in json format."""
import os
import json



class JsonHandler:
    _json_file = ""
    _known_rationals = {}

    @staticmethod
    def _read_data(path):
        with open(path, "rt") as fin:
            data = json.load(fin)
        return data
    
    @staticmethod
    def _write_data(data, path):
        with open(path, "wt") as fout:
            json.dump(data, fout)

    @classmethod
    def set_current(cls, path):
        cls._json_file = os.path.abspath(path)
        data = cls._read_data(cls._json_file)
        cls._known_rationals = {version: [func for func in data[version]] for version in data}

    @classmethod
    def create(cls, path, copy=False):
        """Create new json file.
        
        The file is used for future store/load operations using :func:``JsonHandler.set_current``.
        
        Args:
            path (str): Path of new file.
            copy (bool): Copy data of old file.

        Returns:
            path (str): Absolute path of new file.
        """
        new_path = os.path.abspath(path)

        if not os.path.exists(new_path):
            with open(new_path, "x"): pass

        if copy:
            data = cls._read_data(cls._json_file)
            cls._write_data(data, new_path)

        cls.set_current(new_path)
        return new_path

    @classmethod
    def is_stored(cls, version, degrees, name):
        version = f"{version}{degrees[0]}/{degrees[1]}"
        return (version in cls._known_rationals) and (name in cls._known_rationals[version])

    @classmethod
    def load(cls, version, degrees, name):
        data = cls._read_data(cls._json_file)
        version = f"{version}{degrees[0]}/{degrees[1]}"
        return data[version][name]["numerator"], data[version][name]["denominator"]
    
    @classmethod
    def store(cls, version, degrees, name, numerator, denominator):
        # numerator, denominator must be lists
        version = f"{version}{degrees[0]}/{degrees[1]}"
        data = cls._read_data(cls._json_file)
        
        if version not in data:
            data[version] = {}
            cls._known_rationals[version] = []
        data[version][name] = {"numerator": numerator, "denominator": denominator}

        cls._write_data(data, cls._json_file)
        cls._known_rationals[version].append(name)
        