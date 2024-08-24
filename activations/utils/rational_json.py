"""Provides functions to store and load Rationals in json format."""
import os
import json



class JsonHandler:
    """Manages read/write operations on a json file.
    
    Managed json files have the following format:
    
    .. code-block::

        {<version><N>/<M>
            {<func>
                {
                    "numerator": <weights>,
                    "denominator": <weights> 
                }
            }
        }
    """

    _json_file = ""
    _known_rationals = {}

    @staticmethod
    def _read_data(path):
        """Reads a json file.
        
        Args:
            path (str or pathlike):
                Path of file.
                
        Returns:
            dict:
                The key-value pairs read.
        """
        if os.path.getsize(path) == 0:
            return {}
        
        with open(path, "rt") as fin:
            data = json.load(fin)
        return data
    
    @staticmethod
    def _write_data(data, path):
        """Writes to a json file.
        
        Args:
            data (dict):
                Data to write.
            
            path (str or pathlike):
                Path of destination file.
        """
        with open(path, "wt") as fout:
            json.dump(data, fout)

    @classmethod
    def set_current(cls, path):
        """Sets a file to be used by future store/load operations.
        
        Args:
            path (str or pathlike):
                Destination file.
        """
        cls._json_file = os.path.abspath(path) 
        data = cls._read_data(cls._json_file)
        cls._known_rationals = {version: [func for func in data[version]] for version in data}

    @classmethod
    def create(cls, path, copy=False):
        """Creates a new json file.
        
        The file is used for future store/load operations.
        
        Args:
            path (str or pathlike):
                Path of new file.
            copy (bool):
                Copy data of old file to new file.

        Returns:
            str:
                Absolute path of new file.
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
        """Checks if rational is stored in json."""
        version = f"{version}{degrees[0]}/{degrees[1]}"
        return (version in cls._known_rationals) and (name in cls._known_rationals[version])

    @classmethod
    def load(cls, version, degrees, name):
        """Loads weights of rational.
        
        Args:
            version (str):
                Version of :class:`~activations.torch.rationals.rationals.Rational`.

            degrees (tuple(int, int)):
                Degrees of numerator and denominator.

            name (str):
                Name of approximated function.

        Returns:
            numerator (list(float)):
                The weights for numerator.

            denominator (list(float)):
                The weights for denominator.
        """
        data = cls._read_data(cls._json_file)
        version = f"{version}{degrees[0]}/{degrees[1]}"
        return data[version][name]["numerator"], data[version][name]["denominator"]
    
    @classmethod
    def store(cls, version, degrees, name, numerator, denominator):
        """Stores numerator/denominator weights.
        
        Args:
            version (str):
                The version of rational.
                
            degrees (str):
                The degrees of the polynominals.
                
            name (str):
                The name of the approximated function.
                
            numerator, denominator (list(float)):
                The weights to store.
        """
        # numerator, denominator must be lists
        version = f"{version}{degrees[0]}/{degrees[1]}"
        data = cls._read_data(cls._json_file)
        
        if version not in data:
            data[version] = {}
            cls._known_rationals[version] = []
        data[version][name] = {"numerator": numerator, "denominator": denominator}

        cls._write_data(data, cls._json_file)
        cls._known_rationals[version].append(name)
        