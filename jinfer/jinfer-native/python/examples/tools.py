"""Tool calling: a function is a tool. Its signature is the schema, its docstring the description;
a conversation runs the calls the model makes and keeps going until it answers in words."""
import jinfer

def weather(city: str) -> dict:
    """Current weather for a city."""
    print(f"-> weather({city!r})")
    return {"city": city, "celsius": 18, "sky": "clear"}

with jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0") as model:
    conversation = model.conversation(tools=[weather], max_tokens=64, temperature=0)
    print(conversation.chat("What's the weather in Zurich? Use the tool.").text)

    # the single-shot form leaves the call to you
    reply = model.chat("What's the weather in Zurich?", tools=[weather], tool_choice="required", max_tokens=64, temperature=0)
    for call in reply.tool_calls:
        print(f"model asked for {call.name}({call.arguments})")
