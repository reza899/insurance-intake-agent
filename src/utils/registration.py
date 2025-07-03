import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId

from config.settings import settings
from src.database.mongodb import mongodb
from src.models.insurance import CarRegistration, Customer, RegistrationResponse
from src.utils.exceptions import RegistrationError, RegistrationNotFoundError, RegistrationSaveError

logger = logging.getLogger(__name__)


class RegistrationService:
    """Handles registration database operations."""

    @staticmethod
    async def save_registration(customer: Customer, car: CarRegistration, duplicates: List[Dict[str, Any]]) -> str:
        """Save registration to database."""
        try:
            collection = mongodb.get_collection(settings.database_collections_config["registrations"])

            registration_doc = {
                "_id": ObjectId(),
                "customer": customer.model_dump(),
                "car": car.model_dump(),
                "created_at": datetime.now(UTC),
                "is_duplicate": len(duplicates) > 0,
                "duplicate_matches": [d["id"] for d in duplicates[:3]],
            }

            result = await collection.insert_one(registration_doc)
            return str(result.inserted_id)

        except Exception as e:
            raise RegistrationSaveError(f"Failed to save registration: {e}")

    @staticmethod
    async def get_registration(registration_id: str) -> Optional[RegistrationResponse]:
        """Get registration by ID."""
        try:
            collection = mongodb.get_collection(settings.database_collections_config["registrations"])
            registration = await collection.find_one({"_id": ObjectId(registration_id)})

            if not registration:
                return None

            return RegistrationResponse(
                id=str(registration["_id"]),
                customer=Customer(**registration["customer"]),
                car=CarRegistration(**registration["car"]),
                created_at=registration["created_at"],
                is_duplicate=registration.get("is_duplicate", False),
                duplicate_matches=registration.get("duplicate_matches", []),
            )

        except Exception as e:
            logger.error(f"Error retrieving registration {registration_id}: {e}")
            raise RegistrationNotFoundError(f"Registration not found: {registration_id}")

    @staticmethod
    async def update_existing_registration(duplicate_id: str, customer: Customer, car: CarRegistration) -> bool:
        """Update existing registration with new details."""
        try:
            collection = mongodb.get_collection(settings.database_collections_config["registrations"])

            update_data = {
                "customer": {
                    "name": customer.name,
                    "birth_date": customer.birth_date,
                    "address": customer.address,
                },
                "car": {
                    "car_type": car.car_type,
                    "manufacturer": car.manufacturer,
                    "year": car.year,
                    "license_plate": car.license_plate,
                },
                "updated_at": datetime.now(UTC).isoformat(),
            }

            result = await collection.update_one({"_id": ObjectId(duplicate_id)}, {"$set": update_data})

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Update registration failed: {e}")
            raise RegistrationError(f"Failed to update registration: {e}")
