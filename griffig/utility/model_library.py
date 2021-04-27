import json
from pathlib import Path
from typing import Union
from urllib.request import urlopen
from tempfile import NamedTemporaryFile
from shutil import unpack_archive

import requests

from .model_data import ModelArchitecture, ModelData


class ModelLibrary:
    remote_url = 'https://griffig.xyz/api/v1/models/'
    tmp_path = Path('/tmp') / 'griffig-models'

    @classmethod
    def load_model_data(cls, name: Union[str, Path]) -> ModelData:
        model_path = name

        if isinstance(model_path, str):
            model_path = cls.tmp_path / name

            if not model_path.exists():
                print(f'Download model file to {model_path}...')

                try:
                    r = requests.get(url=cls.remote_url + name, timeout=1.0)  # [s]
                except requests.exceptions.Timeout as e:
                    raise Exception('Could not search for model, please use a local model via a Path input.') from e

                if r.status_code == 404:
                    raise Exception(f'Model {name} not found!')

                model_remote_url = r.json()['download_path']

                model_path.mkdir(parents=True, exist_ok=True)
                with urlopen(model_remote_url) as zipresp, NamedTemporaryFile() as tfile:
                    tfile.write(zipresp.read())
                    tfile.seek(0)
                    unpack_archive(tfile.name, model_path, format='zip')

            else:
                print(f'Found model file at {model_path}')

        with open(model_path / 'model_data.json', 'r') as read_file:
            model_data = ModelData(**json.load(read_file))
            model_data.path = model_path / name / 'model' / 'data' / 'model'  # model_data.path
            
        return model_data
