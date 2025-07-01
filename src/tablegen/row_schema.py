# src/tablegen/row_schema.py

from pydantic import BaseModel, Field
from typing import Optional

class SpringProgramRow(BaseModel):
    step: int = Field(..., description="Step index starting from 1")
    feed: float = Field(..., description="Feed increment in coils")
    feed_abs: float = Field(..., description="Cumulative feed in coils")
    diameter: Optional[float] = Field(None, description="Wire diameter in mm")
    rotation: Optional[float] = Field(0, description="Rotation angle in degrees")
    v_pitch: Optional[float] = Field(0, description="Vertical pitch in mm")
    h_pitch: Optional[float] = Field(0, description="Horizontal pitch in mm")
    mand_y: Optional[float] = Field(0, description="Mandrel Y movement in mm")
    mand_z: Optional[float] = Field(0, description="Mandrel Z movement in mm")
    funct: Optional[int] = Field(None, description="Machine function code")
    param: Optional[int] = Field(None, description="Optional parameter code")
    speed: Optional[int] = Field(30, description="Machine speed in %")
