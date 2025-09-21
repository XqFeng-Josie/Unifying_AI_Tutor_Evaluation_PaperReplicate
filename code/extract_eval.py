import re
import json
from argparse import ArgumentParser
 
def load_data(dataset_path):
    # Dataset and output file configuration
    with open(dataset_path, "r", encoding="utf-8") as fp:
        json_data = json.load(fp)
    print(f"Loaded {len(json_data)} samples from {dataset_path}")
    return json_data

def extract_feedback_and_result(case_text: str):
    """解析单个包含 'Feedback:' 的段落，返回 {'feedback': str, 'number': int|None}"""
    # 先把 Feedback: ... 的主体取出来（不包含后面的 Score: 行）
    m = re.search(r'Feedback:\s*([\s\S]*?)(?=\n\s*Score:|\Z)', case_text, flags=re.IGNORECASE)
    if not m:
        return None
    fb_part = m.group(1).strip()

    # 提取数字（优先级： [RESULT n], [Score n], [n], 末尾数字, 然后 Score: n）
    num = None
    for pat in (
        r'\[RESULT\]\s*(\d+)',        # [RESULT] 1
        r'\[Score\s*(\d+)\]',        # [Score 3]
        r'\[\s*(\d+)\]',             # [2]
        r'(\d+)\.?\s*$',             # 末尾数字 3 或 1.
    ):
        mm = re.search(pat, fb_part, flags=re.IGNORECASE)
        if mm:
            num = int(mm.group(1))
            break
    if num is None:
        mm = re.search(r'Score:\s*(\d+)', case_text, flags=re.IGNORECASE)
        if mm:
            num = int(mm.group(1))

    # 清理 feedback 文本：移除所有标记形式（[Score n], [RESULT] n, [n], 末尾数字, 以及行尾 Score: n）
    cleaned = fb_part
    cleaned = re.sub(r'\[Score\s*\d+\]\.?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[RESULT\]\s*\d+\.?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[\s*\d+\]\.?\s*', '', cleaned)                     # [2]
    cleaned = re.sub(r'\n\s*Score:\s*\d+\s*$', '', cleaned, flags=re.IGNORECASE)  # trailing Score: 3 line
    # 移除末尾的孤立短数字（一般为标号，如 " ... 3" 或 " ... 1."），但只移除长度<=2的数字避免误删长数字
    cleaned = re.sub(r'\s+\d{1,2}\.?\s*$', '', cleaned)
    # 去掉多余空白与结尾标点（句末保留合理的句号除非是标号）
    cleaned = cleaned.strip()
    cleaned = cleaned.rstrip('.,;:')  # 如果你想保留句号可以去掉这一行

    if cleaned == None or cleaned == "" or num == None:
        print("*"*100)
        print(case_text)
        print("*"*100)

    return {"feedback": cleaned, "number": num}


desiderata = {
    "eval_mistake_identification_result": 1,  # Yes
    "eval_mistake_location_result": 1,        # Yes
    "eval_revealing_answer_result": 3,        # No
    "eval_providing_guidance_result": 1,      # Yes
    "eval_coherent_result": 1,                # Yes
    "eval_actionability_result": 1,           # Yes
    "eval_tutor_tone_result": 1,              # Encouraging
    "eval_humanness_result": 1,               # Yes
}

def evaluate_desiderata(json_data):
    from collections import defaultdict
    evaluation_result = defaultdict(list)
    for d in json_data:
        for key in desiderata:
            if d['eval_result'][key]['number'] == desiderata[key]:
                if key not in evaluation_result:
                    evaluation_result[key] = [0,0]
                evaluation_result[key][0] += 1
            else:
                if key not in evaluation_result:
                    evaluation_result[key] = [0,0]
                evaluation_result[key][1] += 1
    for key, value in evaluation_result.items():
        accuracy = value[0]/(value[0]+value[1])*100.0
        print(f"{key}: {accuracy:.2f}%")
    return evaluation_result
                
def demo():
    # -------------------- 示例 --------------------
    case1 = """Feedback: The tutor's response sounds natural and provides constructive feedback to the student. [RESULT] 1"""
    case2 = """Feedback: The tutor's response does not accurately point to a genuine mistake and its location, which was specifically the multiplication error in the third step. 3"""
    case3 = """Feedback: The tutor did not reveal the final answer, instead providing a constructive feedback to the student. 
    Score: 3"""

    case4 = """Feedback: The tutor's response sounds natural and engaging, as it acknowledges the student's effort while gently correcting their mistake, [RESULT] 1."""
    case5 = """Feedback: The tutor offers correct and relevant guidance by hinting at the importance of double-checking work, but fails to directly address the student's incorrect calculation of 72 times 5. [2]"""
    case6 = """Feedback: The tutor's response does not accurately point to a genuine mistake and its location. [Score 3]"""

    for i, case in enumerate([case1, case2, case3, case4, case5, case6], 1):
        print(f"Case {i}:", extract_feedback_and_result(case))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i","--input_file", type=str, required=True)
    parser.add_argument("-o","--output_file", type=str, required=True)
    args = parser.parse_args()
    json_data = load_data(args.input_file)
    extract_key_lists=[
    'eval_mistake_identification_result',
    'eval_mistake_location_result',
    'eval_revealing_answer_result',
    'eval_providing_guidance_result',
    'eval_coherent_result',
    'eval_actionability_result',
    'eval_tutor_tone_result',
    'eval_humanness_result']
    result = []
    for d in json_data:
        d_new = {}
        for key in extract_key_lists:
            extract_result = extract_feedback_and_result(d[key])
            d_new[key] = extract_result
        result.append({"conversation_id": d["conversation_id"], "eval_result": d_new})
    json.dump(result, open(args.output_file, 'w'), ensure_ascii=False, indent=2)
    evaluate_desiderata(result)