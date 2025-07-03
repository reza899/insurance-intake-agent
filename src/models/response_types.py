from typing import Any, Dict, List, Optional, TypedDict, Union


class AgentResponseData(TypedDict, total=False):
    """Type for agent response dictionary."""

    response: str
    extracted_data: Dict[str, Any]
    missing_fields: List[str]
    status: str
    errors: Optional[List[str]]
    duplicates: Optional[List[Dict[str, Any]]]
    registration_id: Optional[str]
    error: Optional[str]


class DuplicateMatchData(TypedDict, total=False):
    """Type for duplicate match data."""

    id: str
    masked_name: Optional[str]
    masked_birth_date: Optional[str]
    masked_address: Optional[str]
    car_info: Optional[str]
    license_plate: str
    similarity_score: float


ExtractedDataDict = Dict[str, Union[str, int]]
