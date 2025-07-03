import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import requests

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings  # noqa: E402


class InsuranceIntakeUI:
    """Insurance Intake Gradio UI with agent backend."""

    def __init__(self):
        self.api_base_url = os.environ.get("API_BASE_URL", f"http://localhost:{settings.api_port}")
        self.conversation_history = []

    def _call_api(self, endpoint: str, method: str = "GET", json_data=None) -> dict:
        """Make API call with error handling."""
        try:
            url = f"{self.api_base_url}/{endpoint}"
            timeout = settings.default_timeout if method == "GET" else settings.long_timeout

            if method == "GET":
                response = requests.get(url, timeout=timeout)
            else:
                response = requests.post(url, json=json_data, timeout=timeout)

            return {
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def chat_respond(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]], str, str]:
        """Handle chat interaction."""
        # Add user message to history immediately
        history.append([message, "Thinking..."])

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": message})

        # Call agent API
        payload = {"message": message, "conversation_history": self.conversation_history}
        result = self._call_api("chat", "POST", payload)

        if result["success"]:
            raw_response = result["data"].get("response", "I'm sorry, I couldn't process that.")
            formatted_response = self._format_response(raw_response)
            self.conversation_history.append({"role": "assistant", "content": formatted_response})
        else:
            formatted_response = f"Sorry, there was an error: {result.get('error', 'Unknown error')}"

        # Update last message with actual response
        history[-1][1] = formatted_response

        # Return updated values
        session_info = (
            f"**Messages:** {len(self.conversation_history)}"
            if self.conversation_history
            else "No conversation yet"
        )

        return "", history, session_info, self.check_api_status()

    def reset_chat(self) -> Tuple[List, str]:
        """Reset conversation."""
        self.conversation_history = []
        return [], "No conversation yet"

    def check_api_status(self) -> str:
        """Check API health."""
        result = self._call_api("health")
        return "✅ API Online" if result["success"] else "⚠️ API Offline"

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="Insurance Intake Agent", theme=gr.themes.Soft()) as interface:
            # Header
            gr.Markdown("# 🚗 Car Insurance Registration Agent")
            gr.Markdown("Welcome! I'll help you register for car insurance through a simple conversation.")

            # API Status
            api_status = gr.Markdown(self.check_api_status())

            # Main chat interface
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=500)
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your message",
                            placeholder="Type your message here...",
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    clear_btn = gr.Button("Reset Conversation", variant="secondary", size="sm")

                with gr.Column(scale=1):
                    session_info = gr.Markdown("No conversation yet", label="Session Info")

            # Event handlers
            send_btn.click(
                self.chat_respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, session_info, api_status]
            )

            msg.submit(
                self.chat_respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, session_info, api_status]
            )

            clear_btn.click(
                self.reset_chat,
                outputs=[chatbot, session_info]
            )

            # Load API status on start
            interface.load(self.check_api_status, outputs=[api_status])

        return interface

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(**kwargs)

    @staticmethod
    def _format_response(raw_response: str) -> str:
        """Format response with collapsible thinking section."""
        think_match = re.search(r"<think>(.*?)</think>\s*(.*)", raw_response, flags=re.DOTALL)
        if not think_match:
            return raw_response

        thinking_content = think_match.group(1).strip()
        actual_response = think_match.group(2).strip()

        return f"""**Agent Response:**

{actual_response}

<details>
<summary>🤔 View Agent's Thinking Process</summary>

```
{thinking_content}
```
</details>"""


def main():
    """Main function to run the Gradio app."""
    app = InsuranceIntakeUI()
    app.launch(server_name="0.0.0.0", server_port=settings.ui_port, share=False)


if __name__ == "__main__":
    main()
