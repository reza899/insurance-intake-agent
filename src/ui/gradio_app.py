from typing import List

import gradio as gr
import requests


class InsuranceIntakeUI:
    """Insurance Intake Gradio UI with agent backend."""

    def __init__(self):
        # Use localhost when host is 0.0.0.0 (which is for server binding, not client connections)
        self.api_base_url = "http://localhost:8000"
        self.conversation_history = []

    def check_api_health(self) -> bool:
        """Check if API is available."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_agent_response(self, user_message: str) -> str:
        """Get response from the agent backend."""
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})

            # Call agent backend
            payload = {"message": user_message, "conversation_history": self.conversation_history}

            response = requests.post(f"{self.api_base_url}/chat", json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "I'm sorry, I couldn't process that.")

                # Parse thinking and actual response
                formatted_response = InsuranceIntakeUI._format_response_with_thinking(raw_response)

                # Add agent response to conversation history (store the formatted version)
                self.conversation_history.append({"role": "assistant", "content": formatted_response})

                return formatted_response
            else:
                return "I'm having trouble connecting to the system. Please try again."

        except Exception as e:
            return f"Sorry, there was an error: {str(e)}"

    @staticmethod
    def _format_response_with_thinking(raw_response: str) -> str:
        """Format response to show thinking process in collapsible section."""
        import re

        # Check if response contains thinking
        think_match = re.search(r"<think>(.*?)</think>\s*(.*)", raw_response, flags=re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            actual_response = think_match.group(2).strip()

            # Format with collapsible thinking section
            formatted = f"""**Agent Response:**

{actual_response}

<details>
<summary>🤔 View Agent's Thinking Process</summary>

```
{thinking_content}
```
</details>"""
            return formatted
        else:
            # No thinking found, return as is
            return raw_response

    def chat_function(self, message: str, history: List[List[str]]) -> List[List[str]]:
        """Handle chat interaction."""
        agent_response = self.get_agent_response(message)
        history.append([message, agent_response])
        return history

    def get_session_info(self) -> str:
        """Get current session information from agent."""
        if not self.conversation_history:
            return "No data collected yet. Start the conversation!"

        # Get latest agent response for session info
        last_message = self.conversation_history[-1] if self.conversation_history else None
        if last_message and last_message.get("role") == "assistant":
            # Simple display of conversation progress
            return f"**Conversation Progress:**\n\n{len(self.conversation_history)} messages exchanged"

        return "Conversation in progress..."

    def reset_session(self) -> tuple:
        """Reset the conversation history."""
        self.conversation_history = []
        return [], self.get_session_info()

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        with gr.Blocks(title="Insurance Intake Agent") as interface:
            gr.Markdown("# 🚗 Car Insurance Registration Agent")
            gr.Markdown("Welcome! I'll help you register for car insurance through a simple conversation.")

            # Dynamic API status component
            api_status_display = gr.Markdown()

            # Function to update API status
            def update_api_status():
                api_status = self.check_api_health()
                status_text = "✅ API is online" if api_status else "⚠️ API is offline - running in demo mode"
                return status_text

            # Initial status update
            interface.load(update_api_status, outputs=api_status_display)

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversation", height=400)
                    msg = gr.Textbox(label="Your message", placeholder="Type your message here...")

                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Reset Session", variant="secondary")

                with gr.Column(scale=1):
                    session_info = gr.Markdown(self.get_session_info(), label="Session Information")

            # Event handlers
            def respond(message, history):
                # First show user message immediately
                history.append([message, "Thinking..."])
                yield "", history, self.get_session_info(), update_api_status()

                # Get agent response
                agent_response = self.get_agent_response(message)

                # Update with actual response
                history[-1][1] = agent_response
                yield "", history, self.get_session_info(), update_api_status()

            def reset():
                new_history, new_info = self.reset_session()
                return new_history, new_info

            submit_btn.click(respond, [msg, chatbot], [msg, chatbot, session_info, api_status_display])
            msg.submit(respond, [msg, chatbot], [msg, chatbot, session_info, api_status_display])
            clear_btn.click(reset, outputs=[chatbot, session_info])

        interface.launch(**kwargs)


def main():
    """Main function to run the Gradio app."""
    app = InsuranceIntakeUI()
    app.launch(server_name="0.0.0.0", server_port=8501)


if __name__ == "__main__":
    main()
