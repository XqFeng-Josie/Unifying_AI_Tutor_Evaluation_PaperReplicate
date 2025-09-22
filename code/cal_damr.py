import os
import json

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
    def map_revealing_of_the_answer(label):
        label = label.lower().strip()
        if label.startswith("yes") and "correct" in label:
            return 1
        elif label.startswith("yes") and "incorrect" in label:
            return 2
        elif label.startswith("no"):
            return 3
        else:
            return None
    if key == "Revealing_of_the_Answer":
        return map_revealing_of_the_answer(label)
    else:
        map_dict = Tutor_tone_mapping if key == "Tutor_Tone" else Other_rule_mapping
        for key, value in map_dict.items():
            if label.startswith(key):
                return value
        print(label)
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
def evaluate_ordinary_desiderata(data, data_type="All", verbose=False):
    from collections import defaultdict
    evaluation_result = defaultdict(dict)
    for data in MRBenchv1_data:
        d_type= data['Data']
        if data_type !="All" and data_type != d_type:
            print(f"Skip {d_type}")
            continue
        for model, value in data['anno_llm_responses'].items():
            annotation_point = value['annotation_point']
            for k, v in annotation_point.items(): 
                if v is None:
                    if verbose:
                        print(model, k, v)
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


from collections import defaultdict

root_dir = "../data"
MRBenchv1 = os.path.join(root_dir, "MRBench/MRBench_V1.json")
MRBenchv1_data = json.load(open(MRBenchv1))
print("Number of dialogues in MRBenchv1 is ", len(MRBenchv1_data))
# map the annotation label to the desiderata point
MRBenchv1_data_mapped = []
for data in MRBenchv1_data:
    for key, value in data['anno_llm_responses'].items():
        annotation = value['annotation']
        new_annotation = {}
        for k, v in annotation.items():
            new_annotation[k] = map_annotation_label(k,v)
        value['annotation_point'] = new_annotation
    MRBenchv1_data_mapped.append(data)
# print(MRBenchv1_data_mapped[0])
# evaluate the desiderata point
evaluation_result = evaluate_ordinary_desiderata(MRBenchv1_data)
# print the evaluation result
pd_result=print_evaluation_result(evaluation_result)

import pandas as pd
ss = pd.read_csv('../paper/paper_result.csv',sep='\t')
ss['Tutor'] = ss['Tutor'].apply(lambda x: x.replace("*","")+"_paper")
concat_result = pd.concat([ss, pd_result], axis=0)
concat_result = concat_result.sort_values(by='Tutor')
print(concat_result)