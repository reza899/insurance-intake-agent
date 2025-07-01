from datetime import date, datetime
from typing import List

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Customer(BaseModel):
    """Customer information model."""

    name: str = Field(..., min_length=2, max_length=100, description="Customer full name")
    birth_date: str = Field(..., description="Customer birth date in YYYY-MM-DD format")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate customer name."""
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip().title()

    @field_validator("birth_date")
    @classmethod
    def validate_birth_date(cls, v: str) -> str:
        """Validate birth date is reasonable."""
        from datetime import datetime

        try:
            # Parse the date string
            birth_date = datetime.strptime(v, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Birth date must be in YYYY-MM-DD format")

        today = date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

        if age < 18:
            raise ValueError("Customer must be at least 18 years old")
        if age > 120:
            raise ValueError("Invalid birth date")

        return v  # Return the original string


class CarRegistration(BaseModel):
    """Car registration information model."""

    car_type: str = Field(..., min_length=2, max_length=50, description="Type of car")
    manufacturer: str = Field(..., min_length=2, max_length=50, description="Car manufacturer")
    year: int = Field(..., ge=1900, le=2025, description="Manufacturing year")
    license_plate: str = Field(..., min_length=2, max_length=20, description="License plate number")

    @field_validator("car_type", "manufacturer")
    @classmethod
    def validate_text_fields(cls, v: str) -> str:
        """Validate text fields."""
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip().title()

    @field_validator("license_plate")
    @classmethod
    def validate_license_plate(cls, v: str) -> str:
        """Validate license plate format."""
        cleaned = v.strip().upper().replace(" ", "").replace("-", "")
        if len(cleaned) < 2:
            raise ValueError("License plate too short")
        return cleaned

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: int) -> int:
        """Validate manufacturing year."""
        current_year = datetime.now().year
        if v > current_year + 1:
            raise ValueError("Invalid manufacturing year")
        return v


class RegistrationRequest(BaseModel):
    """Complete registration request model."""

    customer: Customer
    car: CarRegistration

    model_config = ConfigDict(json_encoders={date: lambda v: v.isoformat()})


class RegistrationResponse(BaseModel):
    """Registration response model."""

    id: str = Field(..., description="Registration ID")
    customer: Customer
    car: CarRegistration
    created_at: datetime
    is_duplicate: bool = False
    duplicate_matches: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        json_encoders={
            ObjectId: str,
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }
    )
