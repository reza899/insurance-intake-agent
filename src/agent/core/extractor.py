import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from config.settings import settings
from src.models.insurance import CarRegistration, Customer
from src.utils.llm_helpers import create_llm_request_and_get_response

logger = logging.getLogger(__name__)


class DataExtractor:
    """LLM-powered data extractor for insurance registration."""

    @staticmethod
    async def extract_data(message: str, existing_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract data from conversation with explicit field mapping."""
        existing_data = existing_data or {}

        prompt = settings.get_prompt("data_extraction").format(
            message=message,
            existing_data=json.dumps(existing_data)
        )

        response = await create_llm_request_and_get_response(
            prompt=prompt,
            context="You extract insurance data and return only valid JSON."
        )

        if not response:
            logger.error("LLM extraction failed")
            return existing_data

        try:
            # Clean and parse JSON
            clean_response = response.strip().replace('```json', '').replace('```', '').strip()
            extracted = json.loads(clean_response)
            return {**existing_data, **extracted}

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from LLM: {response}")
            return existing_data

    @staticmethod
    def validate_data(data: Dict) -> Tuple[Optional[Customer], Optional[CarRegistration], List[str]]:
        """Validate data using Pydantic models."""
        errors = []
        customer = None
        car = None

        # Validate Customer
        if "customer_name" in data and "birth_date" in data:
            try:
                customer = Customer(
                    name=data["customer_name"],
                    birth_date=data["birth_date"],
                    address=data.get("address")
                )
            except ValidationError as e:
                errors.extend([
                    settings.get_response_template("error_message_template")
                    .format(field=err["loc"][0], message=err["msg"])
                    for err in e.errors()
                ])

        # Validate CarRegistration
        car_fields = ["car_type", "manufacturer", "year", "license_plate"]
        if all(field in data for field in car_fields):
            try:
                car = CarRegistration(
                    car_type=data["car_type"],
                    manufacturer=data["manufacturer"],
                    year=data["year"],
                    license_plate=data["license_plate"],
                )
            except ValidationError as e:
                errors.extend([
                    settings.get_response_template("error_message_template")
                    .format(field=err["loc"][0], message=err["msg"])
                    for err in e.errors()
                ])

        return customer, car, errors

    @staticmethod
    def get_missing_fields(data: Dict[str, Any]) -> List[str]:
        """Get missing required fields from config."""
        required = settings.required_fields
        return [field for field in required if field not in data or not data[field]]
