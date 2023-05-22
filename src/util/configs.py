import json

from pydantic import BaseModel


class DataclassSaveMixin(BaseModel):
    '''Methods for saving a dataclass state'''
    @classmethod
    def load(cls, location: str):
        '''Load Configs'''
        with open(location, 'r', encoding='utf8') as file:
            return cls(**json.loads(file.read()))

    def save(self, location):
        '''Store Configs'''
        with open(location, 'w', encoding='utf8') as file:
            file.write(json.dumps(self.dict(), indent=4))
