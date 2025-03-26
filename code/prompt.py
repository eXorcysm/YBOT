"""

This module contains functions for building the model prompts.

"""

### Importing libraries ###

import sys

### Environment settings ###

sys.path.append("../")

PROMPT = "./data/prompts/character.txt"

### Module functions ###

def build_character_message(card, bot = None, usr = None):
    header  = "### FIRST ###"
    footer  = "### END ###"
    message = card[card.index(header) + len(header) : card.index(footer)].format(
        character = bot, user = usr
    )

    return message.strip()

def build_character_persona(card, bot = None, usr = None):
    header  = "### CHARACTER ###"
    footer  = "### USER ###"
    persona = card[card.index(header) + len(header) : card.index(footer)].format(
        character = bot, user = usr
    )

    return persona.strip()

def build_example_dialogue(card, bot = None, usr = None):
    header  = "### EXAMPLE ###"
    footer  = "### FIRST ###"
    example = card[card.index(header) + len(header) : card.index(footer)].format(
        character = bot, user = usr
    )

    return example.strip()

def build_model_instructions(card, bot = None, usr = None):
    header   = "### INSTRUCT ###"
    footer   = "### CHARACTER ###"
    instruct = card[card.index(header) + len(header) : card.index(footer)].format(
        character = bot, user = usr
    )

    return instruct.strip()

def build_prompt_template(bot, usr):
    instruct, example, first_msg = build_system_prompt(bot, usr)

    if example:
        template = instruct + "\n\n" + example + "\n\n" + first_msg
    else:
        template = instruct + "\n\n" + first_msg

    prompt_template = (
        template + """

If provided, use the following details as context relevant to subsequent conversations:

-----
{context}
-----

User: {query}
Assistant: 
"""
    )

    return prompt_template

def build_scenario(card, bot = None, usr = None):
    header = "### SCENE ###"
    footer = "### EXAMPLE ###"
    scene  = card[card.index(header) + len(header) : card.index(footer)].format(
        character = bot, user = usr
    )

    return scene.strip()


def build_system_prompt(bot, usr):
    with open(PROMPT, encoding = "utf-8") as txt:
        card = txt.read()

    example   = build_example_dialogue(card, bot, usr)
    first_msg = build_character_message(card, bot, usr)

    prompt  = build_model_instructions(card, bot, usr) + "\n\n"
    prompt += build_character_persona(card, bot, usr) + "\n\n"
    prompt += build_user_persona(card, bot, usr) + "\n\n"
    prompt += build_scenario(card, bot, usr)

    return prompt, example, first_msg


def build_user_persona(card, bot = None, usr = None):
    header  = "### USER ###"
    footer  = "### SCENE ###"
    persona = card[card.index(header) + len(header) : card.index(footer)].format(
        character = bot, user = usr
    )

    return persona.strip()

def main():
    """
    Display prompt template used by chatbot.
    """

    template = build_prompt_template(bot = "YBOT", usr = "USER")

    print("\n========== PROMPT TEMPLATE ==========\n")
    print(template)
    print("========== PROMPT TEMPLATE ==========")

if __name__ == "__main__":
    main()
