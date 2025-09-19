"""
Helper functions for AI tutor evaluation inference
Contains prompt formatting and text processing utilities
"""

def MathDial_Prompt(org_prompt, streamdata, org=True):
    """
    Format the MathDial prompt template with conversation history
    
    Args:
        org_prompt (str or list): The template string with {history} placeholder or list of message templates
        streamdata (dict): Data containing conversation_history
        org (bool): If True, use original string mode; if False, use chat message mode
    
    Returns:
        str or list: Formatted prompt string (org=True) or list of message dicts (org=False)
    """
    if org:
        return org_prompt.format(history=streamdata['conversation_history'])
    else:
        org_prompt = org_prompt[0]
        return [
            {"role": "system", "content": org_prompt['system']},
            {"role": "user", "content": org_prompt['user'].format(history=streamdata['conversation_history'])}
        ]

def Bridge_Prompt(org_prompt, streamdata, org=True):
    """
    Format the Bridge prompt template with topic and conversation history
    
    Args:
        org_prompt (str or list): The template string with {topic} and {history} placeholders or list of message templates
        streamdata (dict): Data containing Topic and conversation_history
        org (bool): If True, use original string mode; if False, use chat message mode
    
    Returns:
        str or list: Formatted prompt string (org=True) or list of message dicts (org=False)
    """
    if org:
        return org_prompt.format(topic=streamdata['Topic'], history=streamdata['conversation_history'])
    else:
        org_prompt = org_prompt[0]
        return [
            {"role": "system", "content": org_prompt['system']},
            {"role": "user", "content": org_prompt['user'].format(topic=streamdata['Topic'], history=streamdata['conversation_history'])}
        ]

def safe_cut_at_first_heading(text: str) -> str:
    """
    Cut text at first "###" heading marker (original implementation)
    
    Args:
        text (str): Generated text from the model
    
    Returns:
        str: Text before the first "###" marker, stripped
    """
    return text.split("###", 1)[0].strip()