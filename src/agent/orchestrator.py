import logging
from typing import Dict, List, Optional

from config.settings import settings
from src.agent.core import DataExtractor, DuplicateDetector
from src.models import ConversationHistoryItem
from src.models.insurance import CarRegistration, Customer, RegistrationResponse
from src.models.response_types import AgentResponseData
from src.utils.llm_helpers import create_llm_request_and_get_response
from src.utils.registration import RegistrationService

logger = logging.getLogger(__name__)


class InsuranceAgent:
    """Insurance registration agent powered by LLM decision making."""


    async def process_message(
        self, message: str, conversation_history: List[ConversationHistoryItem] = None
    ) -> AgentResponseData:
        """Process message with LLM making all decisions."""
        try:
            # Check if this is an informational query
            if await InsuranceAgent._is_informational_query(message):
                return await InsuranceAgent._handle_informational_query(message)

            # Check if last interaction was a completed registration
            if InsuranceAgent._was_last_registration_completed(conversation_history or []):
                # Start fresh conversation for new registration
                extracted_data = await DataExtractor.extract_data(message, {})
            else:
                # Continue existing conversation
                conversation = InsuranceAgent._build_conversation(message, conversation_history or [])
                extracted_data = await DataExtractor.extract_data(conversation, {})

            # Get current state
            missing_fields = DataExtractor.get_missing_fields(extracted_data)
            customer, car, validation_errors = DataExtractor.validate_data(extracted_data)
            duplicates = await InsuranceAgent._get_duplicates_if_ready(customer, car, missing_fields, validation_errors)

            # Check if this is a duplicate confirmation response
            if await InsuranceAgent._is_duplicate_response(message, conversation_history, duplicates):
                return await InsuranceAgent._handle_duplicate_response(
                    message, extracted_data, customer, car, duplicates
                )

            # Let LLM decide everything and execute
            return await InsuranceAgent._llm_decide_and_execute(
                message, extracted_data, missing_fields, validation_errors, duplicates, customer, car
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            return {
                "response": settings.get_response_template("default_error_response"),
                "extracted_data": {},
                "status": settings.conversation_status_config.get("error", "error"),
                "missing_fields": [],
                "error": str(e)
            }

    @staticmethod
    def _build_conversation(message: str, history: List[ConversationHistoryItem]) -> str:
        """Build conversation text."""
        conversation = "\n".join([f"{item.role}: {item.content}" for item in history])
        return f"{conversation}\nuser: {message}" if conversation else f"user: {message}"

    @staticmethod
    async def _get_duplicates_if_ready(customer: Optional[Customer], car: Optional[CarRegistration],
                                       missing_fields: List[str], validation_errors: List[str]) -> List[Dict]:
        """Get duplicates only if data is complete and valid."""
        if customer and car and not missing_fields and not validation_errors:
            return await DuplicateDetector.find_duplicates(customer, car)
        return []

    @staticmethod
    async def _is_duplicate_response(
        message: str, history: List[ConversationHistoryItem], duplicates: List[Dict]
    ) -> bool:
        """Check if this is a response to duplicate detection."""
        if not history or not duplicates:
            return False

        # Get last assistant message
        last_message = InsuranceAgent._get_last_assistant_message(history)
        if not last_message:
            return False

        # Check if last message mentioned duplicates using config keywords
        duplicate_keywords = settings.duplicate_detection_config.get("context_keywords", [])
        return any(keyword in last_message.lower() for keyword in duplicate_keywords)

    @staticmethod
    async def _handle_duplicate_response(message: str, extracted_data: Dict, customer: Customer,
                                       car: CarRegistration, duplicates: List[Dict]) -> AgentResponseData:
        """Handle user response to duplicate detection using LLM."""
        prompt = settings.get_prompt("duplicate_intent_detection").format(message=message)

        intent = await create_llm_request_and_get_response(prompt, "You determine user intent from their response.")
        intent = intent.strip().upper() if intent else "UNCLEAR"

        try:
            if intent == "UPDATE":
                duplicate_id = duplicates[0]["id"]
                await RegistrationService.update_existing_registration(duplicate_id, customer, car)
                response_text = settings.get_response_template("duplicate_update_confirmation").format(
                    registration_id=duplicate_id,
                    customer_name=customer.name,
                    birth_date=customer.birth_date,
                    year=car.year,
                    manufacturer=car.manufacturer,
                    car_type=car.car_type,
                    license_plate=car.license_plate
                )
                return {
                    "response": response_text,
                    "extracted_data": extracted_data,
                    "status": settings.conversation_status_config.get("completed", "completed"),
                    "missing_fields": [],
                    "registration_id": duplicate_id,
                }
            elif intent == "CREATE":
                registration_id = await RegistrationService.save_registration(customer, car, duplicates)
                response_text = settings.get_response_template("registration_summary").format(
                    registration_id=registration_id,
                    customer_name=customer.name,
                    birth_date=customer.birth_date,
                    year=car.year,
                    manufacturer=car.manufacturer,
                    car_type=car.car_type,
                    license_plate=car.license_plate
                )
                return {
                    "response": response_text,
                    "extracted_data": extracted_data,
                    "status": settings.conversation_status_config.get("completed", "completed"),
                    "missing_fields": [],
                    "registration_id": registration_id,
                }
        except Exception as e:
            logger.error(f"Registration operation failed: {e}")
            return {
                "response": settings.get_response_template("error_fallback"),
                "extracted_data": extracted_data,
                "status": settings.conversation_status_config.get("error", "error"),
                "missing_fields": [],
                "error": str(e)
            }

        # Unclear or failed intent detection
        return {
            "response": settings.get_response_template("clarification_needed"),
            "extracted_data": extracted_data,
            "status": settings.conversation_status_config.get("duplicate_found", "duplicate_found"),
            "missing_fields": [],
            "duplicates": duplicates,
        }

    @staticmethod
    async def _llm_decide_and_execute(
        message: str, extracted_data: Dict, missing_fields: List[str],
        validation_errors: List[str],
        duplicates: List[Dict],
        customer: Optional[Customer],
        car: Optional[CarRegistration]
    ) -> AgentResponseData:
        """LLM decides what to do based on current state."""

        # Handle validation errors
        if validation_errors:
            response_text = settings.get_response_template("validation_error_fallback").format(
                errors=", ".join(validation_errors)
            )
            return {
                "response": response_text,
                "extracted_data": extracted_data,
                "status": settings.conversation_status_config.get("validation_error", "validation_error"),
                "missing_fields": missing_fields,
                "errors": validation_errors,
            }

        # Handle missing fields
        if missing_fields:
            # Use LLM to ask for missing field naturally
            prompt = settings.get_prompt("ask_missing_field").format(
                missing_field=missing_fields[0],
                extracted_data=extracted_data
            )
            response_text = await create_llm_request_and_get_response(
                prompt=prompt, context=settings.get_prompt("system")
            )
            if not response_text:
                response_text = f"Could you please provide your {missing_fields[0]}?"

            return {
                "response": response_text,
                "extracted_data": extracted_data,
                "status": settings.conversation_status_config.get("collecting_data", "collecting_data"),
                "missing_fields": missing_fields,
            }

        # Handle duplicates found
        if duplicates and customer and car:
            # Use generic response to protect privacy
            response_text = (
                "I found an existing registration in our system that appears to be very "
                "similar to your information. Would you like to:\n\n"
                "1. Update the existing registration with your new details\n"
                "2. Create a new separate registration\n"
                "3. Review the existing registration first\n\n"
                "Please let me know which option you prefer."
            )
            return {
                "response": response_text,
                "extracted_data": extracted_data,
                "status": settings.conversation_status_config.get("duplicate_found", "duplicate_found"),
                "missing_fields": [],
                "duplicates": duplicates,
            }

        # Complete registration
        if customer and car:
            try:
                registration_id = await RegistrationService.save_registration(customer, car, [])
                response_text = settings.get_response_template("registration_summary").format(
                    registration_id=registration_id,
                    customer_name=customer.name,
                    birth_date=customer.birth_date,
                    year=car.year,
                    manufacturer=car.manufacturer,
                    car_type=car.car_type,
                    license_plate=car.license_plate
                )
                return {
                    "response": response_text,
                    "extracted_data": extracted_data,
                    "status": settings.conversation_status_config.get("completed", "completed"),
                    "missing_fields": [],
                    "registration_id": registration_id,
                }
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                return {
                "response": settings.get_response_template("error_fallback"),
                "extracted_data": extracted_data,
                "status": settings.conversation_status_config.get("error", "error"),
                "missing_fields": [],
                "error": str(e)
            }

        # Default: continue conversation
        return {
            "response": settings.get_response_template("default_greeting"),
            "extracted_data": extracted_data,
            "status": settings.conversation_status_config.get("collecting_data", "collecting_data"),
            "missing_fields": missing_fields,
        }

    @staticmethod
    def _get_last_assistant_message(history: List[ConversationHistoryItem]) -> Optional[str]:
        """Get the last assistant message for context."""
        for msg in reversed(history):
            if msg.role == "assistant":
                return msg.content
        return None

    @staticmethod
    def _was_last_registration_completed(history: List[ConversationHistoryItem]) -> bool:
        """Check if the last assistant message indicated a completed registration."""
        last_message = InsuranceAgent._get_last_assistant_message(history)
        if not last_message:
            return False

        # Check for completion indicators from config
        completion_phrases = settings.get_config("completion_indicators", [])
        return any(phrase in last_message.lower() for phrase in completion_phrases)

    @staticmethod
    async def _is_informational_query(message: str) -> bool:
        """Check if user is asking for information rather than registering."""
        info_keywords = settings.get_config("informational_keywords", [])
        return any(keyword in message.lower() for keyword in info_keywords)

    @staticmethod
    async def _handle_informational_query(message: str) -> AgentResponseData:
        """Handle informational queries about the registration process."""
        response = settings.get_prompt("informational_response")

        return {
            "response": response,
            "extracted_data": {},
            "status": settings.conversation_status_config.get("informational", "informational"),
            "missing_fields": [],
        }

    @staticmethod
    async def get_registration(registration_id: str) -> Optional[RegistrationResponse]:
        """Get registration by ID."""
        return await RegistrationService.get_registration(registration_id)
