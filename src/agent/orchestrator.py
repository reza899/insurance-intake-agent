from datetime import UTC, datetime
from typing import Dict, List, Optional

from bson import ObjectId

from config.settings import settings
from src.agent.duplicate_detector import DuplicateDetector
from src.agent.extractor import DataExtractor
from src.database.mongodb import mongodb
from src.llm.router import LLMRouter
from src.models import LLMRequest, LLMResponse
from src.models.insurance import CarRegistration, Customer, RegistrationResponse

from .constants import (
    ConversationStatus,
    DatabaseCollections,
    DuplicateKeywords,
    ErrorMessages,
)


class InsuranceAgent:
    """Main orchestrator for insurance registration conversations."""

    def __init__(self):
        """Initialize insurance agent."""
        self.extractor = DataExtractor()
        self.duplicate_detector = DuplicateDetector()
        self.llm_router = LLMRouter()

    async def process_message(self, message: str, conversation_history: List[Dict] = None) -> Dict:
        """Process a user message and return agent response."""
        conversation_history = conversation_history or []

        # Check if this is a response to a duplicate confirmation
        if InsuranceAgent._is_duplicate_confirmation_response(message, conversation_history):
            return await self._handle_duplicate_confirmation(message, conversation_history)

        # Extract data from current message and history
        extracted_data = await self._extract_all_data(message, conversation_history)

        # Check what's missing
        missing_fields = DataExtractor.get_missing_fields(extracted_data)

        # Validate current data
        customer, car, validation_errors = self.extractor.validate_data(extracted_data)

        # Generate appropriate response
        if validation_errors:
            response_text = await self._generate_validation_error_response(validation_errors)
            return {
                "response": response_text,
                "extracted_data": extracted_data,
                "missing_fields": missing_fields,
                "status": ConversationStatus.VALIDATION_ERROR,
                "errors": validation_errors,
            }

        elif missing_fields:
            response_text = await self._generate_follow_up_question(missing_fields, extracted_data)
            return {
                "response": response_text,
                "extracted_data": extracted_data,
                "missing_fields": missing_fields,
                "status": ConversationStatus.COLLECTING_DATA,
            }

        else:
            # All data collected and valid - check for duplicates and complete registration
            return await self._complete_registration(customer, car, extracted_data)

    async def _extract_all_data(self, current_message: str, conversation_history: List[Dict]) -> Dict:
        """Extract data from current message and entire conversation history."""
        # Build complete conversation context
        full_conversation = ""
        for msg in conversation_history:
            if msg.get("role") == "user":
                full_conversation += f"User: {msg.get('content', '')}\n"

        # Add current message
        full_conversation += f"User: {current_message}\n"

        # Extract from entire conversation context at once
        extracted_data = await self.extractor.extract_data(full_conversation, {})

        return extracted_data

    async def _generate_follow_up_question(self, missing_fields: List[str], current_data: Dict) -> str:
        """Generate intelligent, personalized follow-up question."""
        name_greeting = self._get_name_greeting(current_data)
        context_summary = self._build_context_summary(current_data)

        prompt = settings.get_prompt("follow_up_question").format(
            context_summary=context_summary, missing_fields=", ".join(missing_fields), name_greeting=name_greeting
        )

        try:
            response = await self._make_llm_request(prompt)
            return response.content
        except Exception:
            return self._get_fallback_response(missing_fields, name_greeting)

    def _get_name_greeting(self, current_data: Dict) -> str:
        """Extract name greeting from current data."""
        customer_name = current_data.get("customer_name", "")
        return f", {customer_name.split()[0]}" if customer_name else ""

    def _build_context_summary(self, current_data: Dict) -> str:
        """Build context summary from collected data."""
        field_labels = {
            "car_type": "Car type",
            "manufacturer": "Manufacturer",
            "year": "Year",
            "license_plate": "License plate",
            "customer_name": "Customer",
            "birth_date": "Birth date",
        }

        collected_info = [
            f"{field_labels[field]}: {value}"
            for field, value in current_data.items()
            if value and field in field_labels
        ]

        return "; ".join(collected_info) if collected_info else "No information collected yet"

    def _get_fallback_response(self, missing_fields: List[str], name_greeting: str) -> str:
        """Get fallback response when LLM fails."""
        next_field = missing_fields[0] if missing_fields else "information"
        field_name = next_field.replace("_", " ")
        return settings.get_response_template("missing_data_fallback").format(
            name_greeting=name_greeting, field_name=field_name
        )

    async def _generate_validation_error_response(self, errors: List[str]) -> str:
        """Generate response for validation errors."""
        prompt = settings.get_prompt("validation_error").format(errors="; ".join(errors))

        try:
            response = await self._make_llm_request(prompt)
            return InsuranceAgent._clean_response(response.content)
        except Exception:
            return settings.get_response_template("validation_error_fallback").format(errors="; ".join(errors))

    async def _complete_registration(self, customer: Customer, car: CarRegistration, extracted_data: Dict) -> Dict:
        """Complete the registration process."""
        try:
            # Check for duplicates
            duplicates = await self.duplicate_detector.find_duplicates(customer, car)
            is_duplicate = self.duplicate_detector.is_likely_duplicate(duplicates)

            if is_duplicate:
                return self._build_duplicate_response(duplicates, extracted_data)

            # No duplicates - save registration
            registration_id = await InsuranceAgent._save_registration(customer, car, duplicates)
            return self._build_success_response(registration_id, customer, car, extracted_data)

        except Exception as e:
            return {
                "response": settings.get_response_template("error_fallback"),
                "extracted_data": extracted_data,
                "status": ConversationStatus.ERROR,
                "missing_fields": [],
                "error": str(e),
            }

    @staticmethod
    async def _save_registration(customer: Customer, car: CarRegistration, duplicates: List[Dict]) -> str:
        """Save registration to database."""
        try:
            collection = mongodb.get_collection(DatabaseCollections.REGISTRATIONS)

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
            raise Exception(f"{ErrorMessages.REGISTRATION_SAVE_FAILED}: {e}")

    @staticmethod
    async def get_registration(registration_id: str) -> Optional[RegistrationResponse]:
        """Get registration by ID."""
        try:
            collection = mongodb.get_collection(DatabaseCollections.REGISTRATIONS)
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

        except Exception:
            return None

    @staticmethod
    def _clean_response(response: str) -> str:
        """Extract the actual response after Qwen's thinking process."""
        import re

        # Look for content AFTER </think> tags (closed thinking)
        after_think_match = re.search(r"</think>\s*(.+)", response, flags=re.DOTALL)
        if after_think_match:
            cleaned = after_think_match.group(1).strip()
            # Remove surrounding quotes if present
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            elif cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned = cleaned[1:-1]
            return cleaned.strip() if cleaned.strip() else settings.get_response_template("default_greeting")

        # If no closed </think> tag, look for content before <think> (unlikely but possible)
        before_think = re.split(r"<think>", response)[0].strip()
        if before_think:
            return before_think

        # Fallback - just remove think tags entirely and see what's left
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        return cleaned if cleaned else settings.get_response_template("default_greeting")

    async def _make_llm_request(self, prompt: str) -> LLMResponse:
        """Create and send LLM request with standard settings."""
        request = LLMRequest(
            prompt=prompt,
            context=settings.system_prompt,
            temperature=settings.llm_defaults["default_temperature"],
            max_tokens=settings.llm_defaults["default_max_tokens"],
        )
        return await self.llm_router.route_request(request)

    def _build_duplicate_response(self, duplicates: List[Dict], extracted_data: Dict) -> Dict:
        """Build response for duplicate detection scenario."""
        duplicate_info = duplicates[0]

        response_text = settings.get_response_template("duplicate_found").format(
            masked_name=duplicate_info.get("masked_name", "N/A"),
            masked_birth_date=duplicate_info.get("masked_birth_date", "N/A"),
            car_info=duplicate_info.get("car_info", "Unknown vehicle"),
            license_plate=duplicate_info["license_plate"],
            similarity=f"{duplicate_info['similarity_score']:.0%}",
        )

        return {
            "response": response_text,
            "extracted_data": extracted_data,
            "status": ConversationStatus.DUPLICATE_FOUND,
            "missing_fields": [],
            "duplicates": duplicates,
        }

    def _build_success_response(
        self, registration_id: str, customer: Customer, car: CarRegistration, extracted_data: Dict
    ) -> Dict:
        """Build response for successful registration."""
        response_text = settings.get_response_template("registration_summary").format(
            registration_id=registration_id,
            customer_name=customer.name,
            birth_date=customer.birth_date,
            year=car.year,
            manufacturer=car.manufacturer,
            car_type=car.car_type,
            license_plate=car.license_plate,
        )

        return {
            "response": response_text,
            "extracted_data": extracted_data,
            "status": ConversationStatus.COMPLETED,
            "missing_fields": [],
            "registration_id": registration_id,
        }

    async def _proceed_with_registration(self, customer: Customer, car: CarRegistration, extracted_data: Dict) -> Dict:
        """Handle user's decision to proceed with registration despite duplicates."""
        try:
            duplicates = await self.duplicate_detector.find_duplicates(customer, car)
            registration_id = await InsuranceAgent._save_registration(customer, car, duplicates)

            response_text = settings.get_response_template("registration_summary_with_duplicate").format(
                registration_id=registration_id,
                customer_name=customer.name,
                birth_date=customer.birth_date,
                year=car.year,
                manufacturer=car.manufacturer,
                car_type=car.car_type,
                license_plate=car.license_plate,
            )

            return {
                "response": response_text,
                "extracted_data": extracted_data,
                "status": ConversationStatus.COMPLETED,
                "registration_id": registration_id,
            }
        except Exception as e:
            return {
                "response": settings.get_response_template("error_fallback"),
                "status": ConversationStatus.ERROR,
                "error": str(e),
            }

    @staticmethod
    def _is_duplicate_confirmation_response(message: str, conversation_history: List[Dict]) -> bool:
        """Check if the current message is responding to a duplicate confirmation."""
        # Check if the last assistant message mentioned duplicates
        if not conversation_history:
            return False

        last_assistant_message = None
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                last_assistant_message = msg.get("content", "")
                break

        if not last_assistant_message:
            return False

        # Check if last message was about duplicates
        is_duplicate_context = any(
            keyword in last_assistant_message.lower() for keyword in DuplicateKeywords.DUPLICATE_CONTEXT
        )

        # Check if current response is a confirmation (yes/no type)
        is_confirmation = any(word in message.lower() for word in DuplicateKeywords.CONFIRMATION_WORDS)

        return is_duplicate_context and is_confirmation

    async def _handle_duplicate_confirmation(self, message: str, conversation_history: List[Dict]) -> Dict:
        """Handle user's response to duplicate confirmation."""
        message_lower = message.lower()

        # Extract original registration data from conversation history
        extracted_data = await self._extract_all_data("", conversation_history[:-1])  # Exclude current message
        customer, car, _ = self.extractor.validate_data(extracted_data)

        if not customer or not car:
            return {"response": settings.get_response_template("error_fallback"), "status": ConversationStatus.ERROR}

        # Determine user's intent
        wants_to_proceed = any(keyword in message_lower for keyword in DuplicateKeywords.PROCEED_KEYWORDS)
        wants_to_check = any(keyword in message_lower for keyword in DuplicateKeywords.CHECK_KEYWORDS)

        if wants_to_proceed:
            return await self._proceed_with_registration(customer, car, extracted_data)

        elif wants_to_check:
            # User wants to check/review existing registration
            return {
                "response": settings.get_response_template("duplicate_review_response"),
                "extracted_data": extracted_data,
                "status": ConversationStatus.DUPLICATE_REVIEW_REQUESTED,
            }

        else:
            # Unclear response - ask for clarification
            return {
                "response": settings.get_response_template("clarification_needed"),
                "extracted_data": extracted_data,
                "status": ConversationStatus.CLARIFICATION_NEEDED,
            }
