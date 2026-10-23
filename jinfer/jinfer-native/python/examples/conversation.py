"""A conversation keeps its state between turns: each turn costs only its new tokens, a fork
branches the history for free, a rewind drops turns without recomputing anything."""
import jinfer

with jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0") as model:
    conversation = model.conversation(system="You are a terse assistant.", max_tokens=32, temperature=0)
    conversation.chat("My name is Zoltan and I live in Zurich.")
    reply = conversation.chat("Where do I live?")
    print(reply.text, f"({reply.usage.cached_tokens} of {reply.usage.prompt_tokens} prompt tokens came from the cache)")

    branch = conversation.fork()                         # same history, a different future
    print("branch:", branch.chat("And what is my name?").text)
    conversation.rewind(1)                               # back to before the question
    print("rewound to", len(conversation.messages), "messages:", conversation.chat("What is 2+2? Digits only.").text)
