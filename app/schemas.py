from pydantic import BaseModel, Field
from typing import Literal
class TripInput(BaseModel):
    vendor_id:str=Field(..., description="Vendor ID")
    pickup_datetime:str=Field(..., description="Pickup Date and Time, in the format YYYY-MM-DD HH:MM:SS")
    passenger_count:int=Field(..., description="Number of Passengers")
    pickup_longitude:float=Field(..., description="Pickup Longitude")
    pickup_latitude:float=Field(..., description="Pickup Latitude")
    dropoff_longitude:float=Field(..., description="Dropoff Longitude")
    dropoff_latitude:float=Field(..., description="Dropoff Latitude")
    store_and_fwd_flag: Literal["Y", "N"]

