import logging
import re
from typing import Any, Dict, List, Optional

import dspy
from pydantic import ValidationError

from config.settings import settings
from src.models.insurance import CarRegistration, Customer

from .constants import ErrorMessages, LLMProvider, RequiredFields

logger = logging.getLogger(__name__)


class InsuranceExtraction(dspy.Signature):
    """Extract insurance data from customer messages."""

    message = dspy.InputField(desc="Customer message about car insurance")

    # Output fields - only extract if explicitly mentioned
    customer_name = dspy.OutputField(desc="Customer's full name (empty if not mentioned)")
    manufacturer = dspy.OutputField(desc="Car manufacturer like Ford, Toyota (empty if not mentioned)")
    car_type = dspy.OutputField(desc="Car type like Sedan, SUV (empty if not mentioned)")
    year = dspy.OutputField(desc="Car year as number (empty if not mentioned)")
    license_plate = dspy.OutputField(desc="License plate number (empty if not mentioned)")
    birth_date = dspy.OutputField(desc="Birth date as YYYY-MM-DD (empty if not mentioned)")


class DataExtractor:
    """Extract insurance data using DSPy."""

    def __init__(self):
        """Initialize DSPy extractor."""
        try:
            # Initialize DSPy with appropriate provider based on settings
            lm = None

            if settings.use_hf_local:
                # Use HuggingFace local model (not recommended for extraction)
                logger.warning(ErrorMessages.HF_NOT_RECOMMENDED)
                self.extract = None
                return

            # Use primary external provider model for DSPy
            if not settings.ext_provider_model:
                logger.error("External provider model is required for extraction")
                self.extract = None
                return

            # Configure DSPy with OpenAI-compatible API
            lm = dspy.LM(
                model=f"{LLMProvider.OPENAI_MODEL_PREFIX}{settings.ext_provider_model}",
                api_key=settings.ext_provider_api_key or "not-needed",
                api_base=settings.ext_provider_base_url,
            )
            logger.info(f"Using external model {settings.ext_provider_model} for extraction")

            # Configure DSPy with the selected LLM
            dspy.configure(lm=lm)

            # Create extraction module
            self.extract = dspy.ChainOfThought(InsuranceExtraction)
            logger.info("DSPy extractor initialized successfully")

        except Exception as e:
            logger.error(f"{ErrorMessages.DSPY_INIT_FAILED}: {e}")
            self.extract = None

    @staticmethod
    def _parse_dspy_markers(text: str) -> Dict[str, str]:
        """Parse DSPy marker format to extract structured data."""
        # Pattern to match DSPy markers: [[ ## field_name ## ]] value
        pattern = r"\[\[\s*##\s*(\w+)\s*##\s*\]\]\s*([^[]*?)(?=\[\[|\Z)"
        matches = re.findall(pattern, text, re.DOTALL)

        extracted = {}
        for field_name, value in matches:
            cleaned_value = value.strip()
            if cleaned_value and cleaned_value.lower() not in ["empty", "not mentioned", "none", ""]:
                extracted[field_name] = cleaned_value

        return extracted

    async def extract_data(self, message: str, existing_data: Optional[Dict] = None) -> Dict:
        """Extract data from message."""
        existing_data = existing_data or {}

        # Try DSPy extraction if available
        if self.extract:
            try:
                # Run DSPy extraction
                result = self.extract(message=message)

                # Process results - handle both standard attributes and marker format
                extracted: Dict[str, Any] = {}

                # First try to parse as marker format (more robust)
                if hasattr(result, "reasoning") and "[[ ##" in str(result.reasoning):
                    # Parse the reasoning text that contains markers
                    marker_data = self._parse_dspy_markers(str(result.reasoning))
                    extracted.update(marker_data)

                # Also try standard attribute access (fallback)
                if hasattr(result, "customer_name") and result.customer_name.strip():
                    if "customer_name" not in extracted:
                        extracted["customer_name"] = result.customer_name.strip()

                if hasattr(result, "manufacturer") and result.manufacturer.strip():
                    if "manufacturer" not in extracted:
                        extracted["manufacturer"] = result.manufacturer.strip().title()

                if hasattr(result, "car_type") and result.car_type.strip():
                    if "car_type" not in extracted:
                        extracted["car_type"] = result.car_type.strip().title()

                if hasattr(result, "year") and result.year.strip():
                    if "year" not in extracted:
                        try:
                            extracted["year"] = int(result.year.strip())
                        except ValueError:
                            pass

                if hasattr(result, "license_plate") and result.license_plate.strip():
                    if "license_plate" not in extracted:
                        extracted["license_plate"] = result.license_plate.strip().upper()

                if hasattr(result, "birth_date") and result.birth_date.strip():
                    if "birth_date" not in extracted:
                        extracted["birth_date"] = result.birth_date.strip()

                # Clean up extracted data
                cleaned_extracted: Dict[str, Any] = {}
                for key, value in extracted.items():
                    if isinstance(value, str):
                        # Clean up data types and formatting
                        if key == "manufacturer":
                            cleaned_extracted[key] = value.title()
                        elif key == "car_type":
                            cleaned_extracted[key] = value.title()
                        elif key == "license_plate":
                            cleaned_extracted[key] = value.upper()
                        elif key == "year":
                            try:
                                cleaned_extracted[key] = int(value)
                            except ValueError:
                                # Try to extract year from string like "2019"
                                year_match = re.search(r"\b(19|20)\d{2}\b", value)
                                if year_match:
                                    cleaned_extracted[key] = int(year_match.group())
                        else:
                            cleaned_extracted[key] = value
                    else:
                        cleaned_extracted[key] = value

                # Merge with existing data
                merged = {**existing_data, **cleaned_extracted}
                logger.info(f"DSPy extracted: {cleaned_extracted}")
                return merged

            except Exception as e:
                logger.error(f"{ErrorMessages.DSPY_EXTRACTION_FAILED}: {e}")

        # If DSPy is not available or fails, return existing data
        logger.warning(ErrorMessages.DSPY_NOT_AVAILABLE)
        return existing_data

    def validate_data(self, data: Dict) -> tuple[Optional[Customer], Optional[CarRegistration], List[str]]:
        """Validate extracted data."""
        errors = []
        customer = None
        car = None

        # Try to create Customer
        if "customer_name" in data and "birth_date" in data:
            try:
                customer = Customer(name=data["customer_name"], birth_date=data["birth_date"])
            except ValidationError as e:
                errors.extend([f"Customer {err['loc'][0]}: {err['msg']}" for err in e.errors()])

        # Try to create CarRegistration
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
                errors.extend([f"Car {err['loc'][0]}: {err['msg']}" for err in e.errors()])

        return customer, car, errors

    @staticmethod
    def get_missing_fields(data: Dict) -> List[str]:
        """Get missing required fields."""
        required = RequiredFields.ALL
        return [field for field in required if field not in data or not data[field]]

    @staticmethod
    def is_complete(data: Dict) -> bool:
        """Check if all fields are present."""
        return len(DataExtractor.get_missing_fields(data)) == 0
