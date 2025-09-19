system_prompt = """
You are an experienced elementary school math teacher, and you are going to respond to astudent's mistake in a useful and caring way
"""

user_prompt = """
The problem your student is solving is on the topic: {topic}
Conversation History: {history}
"""

assistant_prompt = """
Tutor response (maximum one sentence that is most appropriate given topic and conversation history):
"""


def gen_prompt(streamdata):
    """
    Format the Bridge prompt template with topic and conversation history
    
    Args:
        streamdata (dict): Data containing Topic and conversation_history
    
    Returns:
        str or list: Formatted prompt string (org=True) or list of message dicts (org=False)
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(topic=streamdata['Topic'], history=streamdata['conversation_history'])},
        {"role": "assistant", "content": assistant_prompt}
    ]