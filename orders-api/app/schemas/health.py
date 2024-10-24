from pydantic import BaseModel


class Health(BaseModel):
    name: str
    version_api: str
    version_model: str
