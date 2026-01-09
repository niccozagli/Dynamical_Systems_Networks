from pydantic import BaseModel, ConfigDict, Field

class EulerMaruyamaParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tmin: float = 0.0
    tmax: float = 100.0
    dt: float = Field(0.01, gt=0)

    stats_every: int = Field(1, gt=0)
    write_stats_at_start: bool = True

    state_every: int = Field(1, gt=0)
    write_state_at_start: bool = True
