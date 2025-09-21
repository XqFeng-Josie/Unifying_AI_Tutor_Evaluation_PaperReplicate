import json
import os

'''
This script is used to print the original evaluation result of the MRBenchv1 data.
'''

def map_annotation_label(key, label):
    label = label.lower().strip()
    Tutor_tone_mapping = { 
        "encouraging": 1,
        "neutral": 2,
        "offensive": 3
    }
    Other_rule_mapping = {
        "yes": 1,
        "to some extent": 2,
        "no": 3
    }
    def map_Revealing_of_the_Answer(label):
        if label.startswith("yes"):
            if "correct" in label:
                return 1
            elif "incorrect" in label:
                return 2
            else:
                return -1
        elif label.startswith("no"):
            return 3
        return None
    if key == "Revealing_of_the_Answer":
        return map_Revealing_of_the_Answer(label)
    else:
        map_dict = Tutor_tone_mapping if key == "Tutor_Tone" else Other_rule_mapping
        for key, value in map_dict.items():
            if label.startswith(key):
                return value
        return None


desiderata = {
    "Mistake_Identification": 1,  # Yes
    "Mistake_Location": 1,        # Yes
    "Revealing_of_the_Answer": 3,        # No
    "Providing_Guidance": 1,      # Yes
    "Actionability": 1,           # Yes
    "Coherence": 1,                # Yes
    "Tutor_Tone": 1,              # Encouraging
    "humanlikeness": 1,               # Yes
}

# new_annotation
def evaluate_ordinary_desiderata(MRBenchv1_data):
    from collections import defaultdict
    evaluation_result = defaultdict(dict)
    for data in MRBenchv1_data:
        for model, value in data['anno_llm_responses'].items():
            annotation_point = value['annotation_point']
            for k, v in annotation_point.items(): 
                if v is None:
                    continue
                if v == desiderata[k]:
                    if k not in evaluation_result[model]:
                        evaluation_result[model][k] = [0,0]
                    evaluation_result[model][k][0] += 1
                else:
                    if k not in evaluation_result[model] and v is not None:
                        evaluation_result[model][k] = [0,0]
                    evaluation_result[model][k][1] += 1
    return evaluation_result

def print_evaluation_result(evaluation_result):
    import pandas as pd
    pd_result = []
    columns = []
    for model, value in evaluation_result.items():
        model_result = []
        value = sorted(value.items(), key=lambda x: x[0])
        columns = [k for k, v in value]
        for k, v in value:
            model_result.append((v[0]/(v[0]+v[1] )* 100.0))
        pd_result.append([model] + model_result)  
    columns = ['Tutor'] + columns
    pd_result = pd.DataFrame(pd_result, columns=columns)
    print(pd_result.columns)
    columns_mapping = {
        'Mistake_Identification': 'Mistake_Identification',
        'Mistake_Location': 'Mistake_Location',
        'Revealing_of_the_Answer': 'Revealing_of_the_Answer',
        'Providing_Guidance': 'Providing_Guidance',
        'Actionability': 'Actionability',
        'Coherence': 'Coherence',
        'Tutor_Tone': 'Tutor_Tone',
        'humanlikeness': 'Human-likeness'
    }
    pd_result.rename(columns=columns_mapping, inplace=True)
    pd_result = pd_result[['Tutor', 'Mistake_Identification', 'Mistake_Location', 'Revealing_of_the_Answer', 'Providing_Guidance', 'Actionability', 'Coherence', 'Tutor_Tone', 'Human-likeness']].round(2)
    return pd_result


def get_original_eval(data_type):
    MRBenchv1 = "../data/MRBench/MRBench_V1.json"
    MRBenchv1_data = json.load(open(MRBenchv1))
    MRBenchv1_data_mapped = []
    for data in MRBenchv1_data:
        if data_type != "All" and data['Data'] != data_type:
            continue
        for key, value in data['anno_llm_responses'].items():
            annotation = value['annotation']
            new_annotation = {}
            for k, v in annotation.items():
                new_annotation[k] = map_annotation_label(k,v)
            value['annotation_point'] = new_annotation
        MRBenchv1_data_mapped.append(data)
    evaluation_result = evaluate_ordinary_desiderata(MRBenchv1_data_mapped)
    pd_result=print_evaluation_result(evaluation_result)
    print(pd_result)

if __name__ == "__main__":
    import sys
    data_type  = "All" # MathDial, Bridge, All
    if len(sys.argv) > 1:
        data_type = sys.argv[1]
    get_original_eval(data_type)