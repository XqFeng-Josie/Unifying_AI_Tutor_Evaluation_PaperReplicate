"""
Helper functions for AI tutor evaluation inference
Contains prompt formatting and text processing utilities
Replicates original functionality from ori_helper.py
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

def evaluation_prompt(org_prompt, streamdata):
    """
    Format evaluation prompt with history and response
    (Additional function from original helper)
    
    Args:
        org_prompt (str): Template string with $history and $response placeholders
        streamdata (dict): Data containing conversation_history and result
    
    Returns:
        str: Formatted evaluation prompt
    """
    from string import Template
    return Template(org_prompt).substitute(history=streamdata['conversation_history'], response=streamdata['result'])

def cutting_out_answer(result):
    """
    Extract evaluation scores from result text
    (Additional function from original helper)
    
    Args:
        result (str): Text containing evaluation scores
    
    Returns:
        dict: Dictionary with extracted scores and ratings
    """
    # Define evaluation criteria
    definitions = {
        "mistake_identification": "Has the tutor identified a mistake in a student's response?",
        "mistake_location": "Does the tutor's response accurately point to a genuine mistake and its location?",
        "revealing_answer": "Does the tutor reveal the final answer (whether correct or not)?",
        "providing_guidance": "Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint, examples, and so on?",
        "coherent": "Is the tutor's response logically consistent with the student's previous response?",
        "actionability": "Is it clear from the tutor's feedback what the student should do next?",
        "tutor_tone": "Is the tutor's response encouraging, neutral, or offensive?",
        "humanness": "Does the tutor's response sound natural, rather than robotic or artificial?"
    }
    definitions = tuple(definitions.keys())
    
    # Define point to rating mappings
    point2rate = {
        "mistake_identification_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        },
        "mistake_location_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        },
        "revealing_answer_rubric": {
            1: "Yes (and the revealed answer is correct",
            2: "Yes (but the revealed answer is incorrect)",
            3: "No"
        },
        "providing_guidance_rubric": {
            1: "Yes (guidance is correct and relevant to the mistake)",
            2: "To some extent (guidance is provided but it is fully or partially incorrect or incomplete)",
            3: "No"
        },
        "coherent_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        },
        "actionability_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        },
        "tutor_tone_rubric": {
            1: "Encouraging",
            2: "Neutral",
            3: "Offensive"
        },
        "humanness_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        }
    }
    
    temp = {}
    for d in definitions:
        target_word = f"{d} = "
        score = result.find(target_word)
        try:
            grade = int(result[score + len(target_word)])
        except:
            grade = -1
        temp[f"{d}_point"] = grade
        if grade in (1, 2, 3):
            temp[d] = point2rate[f"{d}_rubric"][grade]
        else:
            temp[d] = "grad not exist"
    return temp
