"""Simple Gradio chat interface for Insurance Intake Agent."""

import os
import re
from typing import List

import gradio as gr
import requests


class InsuranceIntakeUI:
    """Simple Insurance Intake Gradio UI."""

    def __init__(self):
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.session_data = {
            "customer_data": {},
            "car_data": {},
            "registration_complete": False,
        }

    def check_api_health(self) -> bool:
        """Check if API is available."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_agent_response(self, user_message: str) -> str:
        """Get response from the conversational agent."""
        self.extract_data_from_message(user_message)

        if not self.session_data["customer_data"].get("name"):
            return (
                "Hello! I'm here to help you register for car insurance. "
                "Let's start with your full name."
            )

        if not self.session_data["customer_data"].get("birth_date"):
            return (
                "Thank you! Now, could you please provide your birth date? "
                "(Format: YYYY-MM-DD)"
            )

        if not self.session_data["car_data"].get("car_type"):
            return (
                "Great! Now let's talk about your car. What type of car do you have? "
                "(e.g., Sedan, SUV, Hatchback)"
            )

        if not self.session_data["car_data"].get("manufacturer"):
            return "What's the manufacturer of your car? (e.g., Toyota, Ford, BMW)"

        if not self.session_data["car_data"].get("year"):
            return "What year was your car manufactured?"

        if not self.session_data["car_data"].get("license_plate"):
            return "Finally, what's your license plate number?"

        if not self.session_data["registration_complete"]:
            self.session_data["registration_complete"] = True
            return (
                "Perfect! I have all the information I need. "
                "Your car insurance registration is complete!"
            )

        return (
            "Your registration has been completed! "
            "Is there anything else I can help you with?"
        )

    def extract_data_from_message(self, user_message: str) -> None:
        """Extract data from user message (simplified logic)."""
        message_lower = user_message.lower().strip()

        if (
            not self.session_data["customer_data"].get("name")
            and len(user_message.split()) >= 2
        ):
            self.session_data["customer_data"]["name"] = user_message.title()

        elif not self.session_data["customer_data"].get("birth_date"):
            date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"
            match = re.search(date_pattern, user_message)
            if match:
                self.session_data["customer_data"]["birth_date"] = match.group()

        elif not self.session_data["car_data"].get("car_type"):
            car_types = ["sedan", "suv", "hatchback", "coupe", "truck", "van"]
            for car_type in car_types:
                if car_type in message_lower:
                    self.session_data["car_data"]["car_type"] = car_type.title()
                    break

        elif not self.session_data["car_data"].get("manufacturer"):
            manufacturers = [
                "toyota",
                "ford",
                "bmw",
                "honda",
                "nissan",
                "chevrolet",
                "audi",
                "mercedes",
            ]
            for manufacturer in manufacturers:
                if manufacturer in message_lower:
                    self.session_data["car_data"]["manufacturer"] = manufacturer.title()
                    break

        elif not self.session_data["car_data"].get("year"):
            year_pattern = r"\b(19|20)\d{2}\b"
            match = re.search(year_pattern, user_message)
            if match:
                self.session_data["car_data"]["year"] = int(match.group())

        elif not self.session_data["car_data"].get("license_plate"):
            cleaned = re.sub(r"[^a-zA-Z0-9]", "", user_message)
            if len(cleaned) >= 3:
                self.session_data["car_data"]["license_plate"] = cleaned.upper()

    def chat_function(self, message: str, history: List[List[str]]) -> List[List[str]]:
        """Handle chat interaction."""
        agent_response = self.get_agent_response(message)
        history.append([message, agent_response])
        return history

    def get_session_info(self) -> str:
        """Get current session information."""
        info = "**Collected Information:**\n\n"

        if self.session_data["customer_data"]:
            info += "**Customer Data:**\n"
            for key, value in self.session_data["customer_data"].items():
                if value:
                    info += f"- {key.title()}: {value}\n"
            info += "\n"

        if self.session_data["car_data"]:
            info += "**Car Data:**\n"
            for key, value in self.session_data["car_data"].items():
                if value:
                    info += f"- {key.title()}: {value}\n"

        if not self.session_data["customer_data"] and not self.session_data["car_data"]:
            info += "No data collected yet. Start the conversation!"

        return info

    def reset_session(self) -> tuple:
        """Reset the session data."""
        self.session_data = {
            "customer_data": {},
            "car_data": {},
            "registration_complete": False,
        }
        return [], self.get_session_info()

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        api_status = self.check_api_health()
        status_text = (
            "✅ API is online"
            if api_status
            else "⚠️ API is offline - running in demo mode"
        )

        with gr.Blocks(title="Insurance Intake Agent") as interface:
            gr.Markdown("# 🚗 Car Insurance Registration Agent")
            gr.Markdown(
                "Welcome! I'll help you register for car insurance "
                "through a simple conversation."
            )
            gr.Markdown(status_text)

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversation", height=400)
                    msg = gr.Textbox(
                        label="Your message", placeholder="Type your message here..."
                    )

                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Reset Session", variant="secondary")

                with gr.Column(scale=1):
                    session_info = gr.Markdown(
                        self.get_session_info(), label="Session Information"
                    )

            # Event handlers
            def respond(message, history):
                new_history = self.chat_function(message, history)
                return "", new_history, self.get_session_info()

            def reset():
                new_history, new_info = self.reset_session()
                return new_history, new_info

            submit_btn.click(respond, [msg, chatbot], [msg, chatbot, session_info])
            msg.submit(respond, [msg, chatbot], [msg, chatbot, session_info])
            clear_btn.click(reset, outputs=[chatbot, session_info])

        interface.launch(**kwargs)


def main():
    """Main function to run the Gradio app."""
    # Read configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("UI_PORT", "8501"))

    app = InsuranceIntakeUI()
    app.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
