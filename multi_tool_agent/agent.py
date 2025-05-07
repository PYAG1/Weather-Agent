import datetime
import os
import asyncio
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

# Load API key from environment variable
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable. Please set it in a .env file or your environment.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Model constants
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

# Tool definitions
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


def say_hello(name: str = "there") -> str:
    """Provides a simple greeting, optionally addressing the user by name.

    Args:
        name (str, optional): The name of the person to greet. Defaults to "there".

    Returns:
        str: A friendly greeting message.
    """
    print(f"--- Tool: say_hello called with name: {name} ---")
    return f"Hello, {name}!"


def say_goodbye() -> str:
    """Provides a simple farewell message to conclude the conversation."""
    print(f"--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."


# Agent creation functions
def create_greeting_agent():
    """Create and return the greeting agent."""
    try:
        agent = Agent(
            model=MODEL_GEMINI_2_0_FLASH,
            name="greeting_agent",
            instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
                        "Use the 'say_hello' tool to generate the greeting. "
                        "If the user provides their name, make sure to pass it to the tool. "
                        "Do not engage in any other conversation or tasks.",
            description="Handles simple greetings and hellos using the 'say_hello' tool.",
            tools=[say_hello],
        )
        print(f"✅ Agent '{agent.name}' created using model '{agent.model}'.")
        return agent
    except Exception as e:
        print(f"❌ Could not create Greeting agent. Error: {e}")
        return None


def create_farewell_agent():
    """Create and return the farewell agent."""
    try:
        agent = Agent(
            model=MODEL_GEMINI_2_0_FLASH,
            name="farewell_agent",
            instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message. "
                        "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the conversation "
                        "(e.g., using words like 'bye', 'goodbye', 'thanks bye', 'see you'). "
                        "Do not perform any other actions.",
            description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.",
            tools=[say_goodbye],
        )
        print(f"✅ Agent '{agent.name}' created using model '{agent.model}'.")
        return agent
    except Exception as e:
        print(f"❌ Could not create Farewell agent. Error: {e}")
        return None


def create_root_agent(greeting_agent, farewell_agent):
    """Create and return the root agent with sub-agents."""
    if not greeting_agent or not farewell_agent:
        print("❌ Cannot create root agent because one or more sub-agents are missing.")
        return None

    try:
        root_agent = Agent(
            name="weather_agent_team",
            model=MODEL_GEMINI_2_0_FLASH,
            description="The main coordinator agent. Handles weather requests and delegates greetings/farewells to specialists.",
            instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
                        "Use the 'get_weather' tool ONLY for specific weather requests (e.g., 'weather in London'). "
                        "You have specialized sub-agents: "
                        "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
                        "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
                        "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
                        "If it's a weather request, handle it yourself using 'get_weather'. "
                        "For anything else, respond appropriately or state you cannot handle it.",
            tools=[get_weather],
            sub_agents=[greeting_agent, farewell_agent]
        )
        print(f"✅ Root Agent '{root_agent.name}' created with sub-agents: {[sa.name for sa in root_agent.sub_agents]}")
        return root_agent
    except Exception as e:
        print(f"❌ Could not create Root agent. Error: {e}")
        return None


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            # You can uncomment the line below to see *all* events during execution
            # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}")

            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
    except Exception as e:
        final_response_text = f"Error during agent execution: {e}"

    print(f"<<< Agent Response: {final_response_text}")


async def run_team_conversation():
    """Run a conversation with the agent team."""
    print("\n--- Testing Agent Team Delegation ---")
    
    # Create agents
    greeting_agent = create_greeting_agent()
    farewell_agent = create_farewell_agent()
    root_agent = create_root_agent(greeting_agent, farewell_agent)
    
    if not root_agent:
        print("Cannot run conversation without a properly configured root agent.")
        return
    
    # Set up session and runner
    session_service = InMemorySessionService()
    app_name = "weather_tutorial_agent_team"
    user_id = "user_1_agent_team"
    session_id = "session_001_agent_team"
    
    session = session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    print(f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'")
    
    runner = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service
    )
    print(f"Runner created for agent '{root_agent.name}'.")
    
    # Run conversation
    await call_agent_async("Hello there!", runner, user_id, session_id)
    await call_agent_async("What is the weather in New York?", runner, user_id, session_id)
    await call_agent_async("Thanks, bye!", runner, user_id, session_id)


# Main execution block
def main():
    """Main function to run the agent conversation."""
    try:
        asyncio.run(run_team_conversation())
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")


if __name__ == "__main__":
    main()