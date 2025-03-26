"""

This module is for running unit tests on the AI agent.

"""

### Module functions ###

def pretty_print_stream_chunk(stream):
    """
    Display agent thought process.
    """

    for node, updates in stream.items():
        print(f"Update from node: {node}")

        if "messages" in updates:
            updates["messages"][-1].pretty_print()
        else:
            print(updates)

        print("\n")

def recall_memories(stream):
    """
    Return memories recalled by agent.
    """

    memories = ""

    for node, updates in stream.items():
        if node == "load_memories":
            memories += "\n".join(updates["recall_memories"])

    return memories

def run_tests(graph):
    """
    Run unit tests on agent.
    """

    # Specify user_id and thread_id to save memories for given user.
    convo_config = {"configurable" : {"user_id" : "1", "thread_id" : "1"}}

    for chunk in graph.stream({"messages" : [("user", "My name is Jim.")]}, config = convo_config):
        pretty_print_stream_chunk(chunk)

    for chunk in graph.stream({"messages" : [("user", "I love pizza.")]}, config = convo_config):
        pretty_print_stream_chunk(chunk)

    for chunk in graph.stream(
        {"messages" : [("user", "Yes -- pepperoni!")]}, config = convo_config
    ):
        pretty_print_stream_chunk(chunk)

    for chunk in graph.stream(
        {"messages" : [("user", "I also just moved to Toronto. You must remember that!")]},
        config = convo_config
    ):
        pretty_print_stream_chunk(chunk)

    for chunk in graph.stream(
        {"messages" : [("user", "Where did I just say I am?")]}, config = convo_config
    ):
        pretty_print_stream_chunk(chunk)

    # Change thread_id to indicate different chat session.
    convo_config = {"configurable" : {"user_id" : "1", "thread_id" : "2"}}

    for chunk in graph.stream({"messages" : [("user", "What is my name?")]},
        config = convo_config
    ):
        pretty_print_stream_chunk(chunk)

    for chunk in graph.stream(
        {"messages" : [("user", "Where should I go for dinner?")]}, config = convo_config
    ):
        pretty_print_stream_chunk(chunk)

    for chunk in graph.stream(
        {"messages" : [("user", "What's the address for Blaze Pizza downtown?")]},
        config = convo_config
    ):
        pretty_print_stream_chunk(chunk)
