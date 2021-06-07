import glob
import json
from pathlib import Path
from shutil import unpack_archive
from tempfile import NamedTemporaryFile
from typing import Union
from urllib.request import urlopen

import requests

from .model_data import ModelArchitecture, ModelData


class ModelLibrary:
    remote_url = 'https://griffig.xyz/api/v1/models/'
    tmp_path = Path('/tmp') / 'griffig' / 'models'

    @classmethod
    def get_model_version_from_name(cls, name: str) -> int:
        return int(name.split('-')[-1][1:])

    @classmethod
    def load_model_data(cls, name: Union[str, Path]) -> ModelData:
        model_path = name

        if isinstance(model_path, str):
            matched_model_globs = glob.glob(str(cls.tmp_path / f'{name}-v*'))
            if not matched_model_globs:
                try:
                    r = requests.get(url=cls.remote_url + name, timeout=1.0)  # [s]
                except requests.exceptions.Timeout as e:
                    raise Exception('Could not search for model, please use a local model via a Path input.') from e

                if r.status_code == 404:
                    raise Exception(f'Model {name} not found!')

                model_info = r.json()
                print(f"Download model {model_info['name']} version {model_info['version']} to {model_path}...")

                model_path = cls.tmp_path / f"{model_info['name']}-v{model_info['version']}"
                model_path.mkdir(parents=True, exist_ok=True)

                model_remote_url = model_info['download_path']
                with urlopen(model_remote_url) as zipresp, NamedTemporaryFile() as tfile:
                    tfile.write(zipresp.read())
                    tfile.seek(0)
                    unpack_archive(tfile.name, model_path, format='zip')

            else:
                matched_model_globs.sort(key=cls.get_model_version_from_name, reverse=True)
                version = cls.get_model_version_from_name(matched_model_globs[0])
                model_path = Path(matched_model_globs[0])
                print(f'Found model {name} version {version} file at {model_path}')

        if isinstance(name, Path):
            name = name.name

        with open(model_path / 'model_data.json', 'r') as read_file:
            model_data = ModelData(**json.load(read_file))
            model_data.path = model_path / name / 'model' / 'data' / 'model'  # model_data.path

        return model_data
