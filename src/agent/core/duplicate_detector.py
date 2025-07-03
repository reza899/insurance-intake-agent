import logging
from typing import Any, Dict, List

from config.settings import settings
from src.database.mongodb import mongodb
from src.models.insurance import CarRegistration, Customer
from src.utils.exceptions import DuplicateDetectionError
from src.utils.llm_helpers import create_llm_request_and_get_response

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """LLM-powered duplicate detection."""

    @staticmethod
    async def find_duplicates(customer: Customer, car: CarRegistration) -> List[Dict[str, Any]]:
        """Find potential duplicates using LLM comparison."""
        try:
            collection = mongodb.get_collection(settings.database_collections_config["registrations"])
            existing_registrations = await collection.find({}).to_list(length=None)

            if not existing_registrations:
                return []

            duplicates = []
            threshold = settings.duplicate_detection_config.get("similarity_threshold", 0.85)

            for registration in existing_registrations:
                score = await DuplicateDetector._get_similarity_score(customer, car, registration)
                if score >= threshold:
                    duplicates.append({
                        "id": str(registration["_id"]),
                        "similarity_score": score,
                        "customer_name": registration["customer"]["name"],
                        "birth_date": registration["customer"]["birth_date"],
                        "license_plate": registration["car"]["license_plate"],
                        "car_info": (
                            f"{registration['car']['year']} "
                            f"{registration['car']['manufacturer']} "
                            f"{registration['car']['car_type']}"
                        ),
                        "created_at": registration["created_at"]
                    })

            return sorted(duplicates, key=lambda x: x["similarity_score"], reverse=True)

        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            raise DuplicateDetectionError(f"Failed to find duplicates: {e}")

    @staticmethod
    async def _get_similarity_score(customer: Customer, car: CarRegistration, existing: Dict[str, Any]) -> float:
        """Get similarity score using LLM."""
        new_data = f"Name: {customer.name}, Birth: {customer.birth_date}, Plate: {car.license_plate}"
        existing_data = (
            f"Name: {existing['customer']['name']}, "
            f"Birth: {existing['customer']['birth_date']}, "
            f"Plate: {existing['car']['license_plate']}"
        )

        prompt = settings.get_prompt("duplicate_comparison").format(
            new_registration=new_data,
            existing_registration=existing_data
        )

        response = await create_llm_request_and_get_response(prompt=prompt)

        if response:
            import re
            score_match = re.search(r"(\d+\.?\d*)", response)
            if score_match:
                return max(0.0, min(1.0, float(score_match.group(1))))

        return 0.0
