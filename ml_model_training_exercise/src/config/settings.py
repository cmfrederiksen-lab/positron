from pydantic import BaseModel, Field
from typing import Optional

class DataConfig(BaseModel):
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    random_state: int = 42

class RandomForestConfig(BaseModel):
    n_estimators: int = Field(default=100, ge=10, le=500)
    # Using Optional[int] instead of int | None for Python 3.9 compatibility
    max_depth: Optional[int] = Field(default=None, ge=1)

class LogisticRegressionConfig(BaseModel):
    C: float = Field(default=1.0, gt=0.0)
    max_iter: int = Field(default=1000, ge=100)

class PipelineConfig(BaseModel):
    data: DataConfig = DataConfig()
    rf: RandomForestConfig = RandomForestConfig()
    lr: LogisticRegressionConfig = LogisticRegressionConfig()

config = PipelineConfig()
