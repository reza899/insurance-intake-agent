import json
import logging
from datetime import date
from typing import Dict, List

from rapidfuzz import fuzz

from config.settings import settings
from src.database.mongodb import mongodb
from src.llm.router import LLMRouter
from src.models import LLMRequest
from src.models.insurance import CarRegistration, Customer

from .constants import DatabaseCollections, ErrorMessages

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """Detects potential duplicate customer registrations."""

    def __init__(self):
        """Initialize duplicate detector."""
        self.config = settings.duplicate_detection_config
        self.threshold = self.config.get("similarity_threshold", 0.85)
        self.weights = self.config.get("weights", {"name": 0.30, "birth_date": 0.30, "license_plate": 0.40})
        self.llm_router = LLMRouter()

    async def find_duplicates(self, customer: Customer, car: CarRegistration) -> List[Dict]:
        """Find potential duplicate registrations."""
        try:
            # Get all existing registrations
            collection = mongodb.get_collection(DatabaseCollections.REGISTRATIONS)
            existing_registrations = await collection.find({}).to_list(length=None)

            if not existing_registrations:
                return []

            potential_duplicates = []

            for registration in existing_registrations:
                # Check for exact license plate match first (same car = duplicate)
                if registration.get("car", {}).get("license_plate", "").upper() == car.license_plate.upper():
                    # Same license plate = definite duplicate
                    potential_duplicates.append(
                        {
                            "id": str(registration["_id"]),
                            "similarity_score": 1.0,  # Perfect match for same car
                            "customer_name": registration["customer"]["name"],
                            "birth_date": registration["customer"]["birth_date"],
                            "license_plate": registration["car"]["license_plate"],
                            "created_at": registration["created_at"],
                            # Masked versions for display
                            "masked_name": DuplicateDetector._mask_name(registration["customer"]["name"]),
                            "masked_birth_date": DuplicateDetector._mask_birth_date(
                                registration["customer"]["birth_date"]
                            ),
                            "car_info": (
                                f"{registration['car'].get('year', '')} "
                                f"{registration['car'].get('manufacturer', '')} "
                                f"{registration['car'].get('car_type', '')}"
                            ),
                        }
                    )
                else:
                    # Try LLM-powered similarity for other potential duplicates
                    similarity_score = await self._intelligent_similarity_check(customer, car, registration)

                    if similarity_score >= self.threshold:
                        potential_duplicates.append(
                            {
                                "id": str(registration["_id"]),
                                "similarity_score": similarity_score,
                                "customer_name": registration["customer"]["name"],
                                "birth_date": registration["customer"]["birth_date"],
                                "license_plate": registration["car"]["license_plate"],
                                "created_at": registration["created_at"],
                                # Masked versions for display
                                "masked_name": DuplicateDetector._mask_name(registration["customer"]["name"]),
                                "masked_birth_date": DuplicateDetector._mask_birth_date(
                                    registration["customer"]["birth_date"]
                                ),
                                "car_info": (
                                    f"{registration['car'].get('year', '')} "
                                    f"{registration['car'].get('manufacturer', '')} "
                                    f"{registration['car'].get('car_type', '')}"
                                ),
                            }
                        )

            # Sort by similarity score (highest first)
            potential_duplicates.sort(key=lambda x: x["similarity_score"], reverse=True)

            return potential_duplicates

        except Exception as e:
            # If database fails, assume no duplicates
            logger.error(f"{ErrorMessages.DUPLICATE_DETECTION_FAILED}: {e}")
            return []

    def _calculate_similarity(self, customer: Customer, car: CarRegistration, existing: Dict) -> float:
        """Calculate overall similarity score between registrations."""
        scores = {}

        # Name similarity
        if "customer" in existing and "name" in existing["customer"]:
            scores["name"] = fuzz.ratio(customer.name.lower(), existing["customer"]["name"].lower()) / 100.0
        else:
            scores["name"] = 0.0

        # Birthdate similarity (exact match or close)
        if "customer" in existing and "birth_date" in existing["customer"]:
            existing_date = existing["customer"]["birth_date"]
            if isinstance(existing_date, str):
                try:
                    existing_date = date.fromisoformat(existing_date)
                except (ValueError, TypeError):
                    existing_date = None

            if existing_date and existing_date == customer.birth_date:
                scores["birth_date"] = 1.0
            else:
                scores["birth_date"] = 0.0
        else:
            scores["birth_date"] = 0.0

        # License plate similarity
        if "car" in existing and "license_plate" in existing["car"]:
            scores["license_plate"] = (
                fuzz.ratio(car.license_plate.upper(), existing["car"]["license_plate"].upper()) / 100.0
            )
        else:
            scores["license_plate"] = 0.0

        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0

        for field, score in scores.items():
            weight = self.weights.get(field, 0.0)
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    async def _intelligent_similarity_check(self, customer: Customer, car: CarRegistration, existing: Dict) -> float:
        """Use LLM to intelligently assess similarity between registrations."""
        try:
            new_registration = {
                "name": customer.name,
                "birth_date": str(customer.birth_date),
                "license_plate": car.license_plate,
                "car": f"{car.year} {car.manufacturer} {car.car_type}",
            }

            existing_car_info = (
                f"{existing.get('car', {}).get('year', '')} "
                f"{existing.get('car', {}).get('manufacturer', '')} "
                f"{existing.get('car', {}).get('car_type', '')}"
            )
            existing_registration = {
                "name": existing.get("customer", {}).get("name", ""),
                "birth_date": str(existing.get("customer", {}).get("birth_date", "")),
                "license_plate": existing.get("car", {}).get("license_plate", ""),
                "car": existing_car_info,
            }

            comparison_prompt = settings.get_prompt("duplicate_comparison").format(
                new_registration=json.dumps(new_registration, indent=2),
                existing_registration=json.dumps(existing_registration, indent=2),
            )

            request = LLMRequest(
                prompt=comparison_prompt,
                context=None,
                temperature=settings.llm_defaults["default_temperature"],
                max_tokens=settings.llm_defaults["default_max_tokens"],
            )

            response = await self.llm_router.route_request(request)

            # Extract numeric score from response
            import re

            score_match = re.search(r"(\d+\.?\d*)", response.content)
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is within valid range
                return max(0.0, min(1.0, score))

        except Exception:
            pass

        # Fallback to traditional similarity calculation
        return self._calculate_similarity(customer, car, existing)

    @staticmethod
    def _mask_name(name: str) -> str:
        """Mask name to protect privacy while showing pattern."""
        if not name:
            return ""

        parts = name.strip().split()
        if len(parts) == 1:
            # Single name: show first letter + asterisks
            return f"{parts[0][0]}***"
        else:
            # Multiple parts: show first letter of each + asterisks
            masked_parts = []
            for part in parts:
                if part:
                    masked_parts.append(f"{part[0]}***")
            return " ".join(masked_parts)

    @staticmethod
    def _mask_birth_date(birth_date) -> str:
        """Mask birth date to show only year."""
        try:
            if isinstance(birth_date, str):
                # Extract year from date string (YYYY-MM-DD format)
                year = birth_date.split("-")[0]
                return f"{year}-**-**"
            elif hasattr(birth_date, "year"):
                # Date object
                return f"{birth_date.year}-**-**"
            else:
                return "****-**-**"
        except (ValueError, TypeError):
            return "****-**-**"

    def is_likely_duplicate(self, duplicates: List[Dict]) -> bool:
        """Check if there are likely duplicates above threshold."""
        return len(duplicates) > 0 and duplicates[0]["similarity_score"] >= self.threshold
