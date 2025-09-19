system_prompt = """
You are an experienced middle school math teacher, and you are going to respond to astudent's mistake in a useful and caring way
"""

user_prompt = """
The conversation history of the problem your student is solving is: {history}
"""

assistant_prompt = """
Tutor response (maximum one sentence that aligns most appropriately with conversation hisotry):
"""



def gen_prompt(streamdata):
    """
    Format the MathDial prompt template with conversation history
    
    Args:
        streamdata (dict): Data containing conversation_history    
    Returns:
        str or list: Formatted prompt string (org=True) or list of message dicts (org=False)
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(history=streamdata['conversation_history'])},
        {"role": "assistant", "content": assistant_prompt}
    ]