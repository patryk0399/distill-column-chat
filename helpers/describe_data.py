from __future__ import annotations
import pandas as pd
import math
from typing import Optional, TypedDict

data = {
    'A': [1, 2, 3, 4, 6],
    'B': [5, 4, 3, 2, 1],
    'C': [2, 3, 4, 5, 6]
}

df = pd.DataFrame(data)

desc = df.describe()
#print(desc)
desc.to_json("helpers/jsons/data_base.jsonl", orient="columns", indent=0) 
# For inference set indent = 0 because it does not use any additional characters
# for "pretty printing". 

data = {
    'A': [1, 200, 3, 4, 500],
    'B': [5, 4, 300, 299, 1],
    'C': [2, 334, 41000, 56, 6]
}

df = pd.DataFrame(data)
desc = df.describe()
#print(desc)
desc.to_json("helpers/jsons/data_test.jsonl", orient="columns", indent=0)
# For inference set indent = 0 because it does not use any additional characters
# for "pretty printing". 

desc = df = pd.read_json("helpers/jsons/data_test.jsonl")

for colname in desc:
    for idx, val in zip(desc[colname].index, desc[colname].values):
        print("index: ", idx,", values: ", val)


class Comparison(TypedDict):
    multiplier: Optional[float]
    percent_of_y: Optional[float]
    percent_change_from_y: Optional[float]


def compare_x_to_y(x: float, y: float) -> Comparison:
    if y == 0:
        # avoid division by 0
        message = "<Can't compare because y = 0.>"
        return ValueError(message)
        
    mult = x / y
    return {
        "multiplier": mult,
        "percent_of_y": f"""{mult * 100}%""",
        "percent_change_from_y": ((x - y) / y) * 100,
    }

desc1 = df = pd.read_json("helpers/jsons/data_base.jsonl")
desc2 = df = pd.read_json("helpers/jsons/data_test.jsonl")

final = {}
final_df = pd.DataFrame()
for colname1, colname2 in zip(desc1,desc2):
    descriptions = []
    for (metric1, val1), (metric2, val2) in zip(zip(desc1[colname1].index, desc1[colname1].values), zip(desc2[colname2].index, desc2[colname2].values)):
    
        comparison = compare_x_to_y(val2, val1)
        if(comparison["multiplier"] > 1):
            x,y = "larger", "increase"
        elif(comparison["multiplier"] == 1):
            x,y = "unchanged", "no change"
        elif(comparison["multiplier"] < 1): 
            x,y = "smaller", "decrease"
        else:
            print("Error")
            continue

        if(x in ["larger", "smaller"]):
            text = f"""
                For
                desc1 for the {colname1} the {metric1} is {val1}
                compared to
                desc2 for the {colname2} the {metric2} is {val2}.

                Comparing both, 
            
                the {metric2} of {val2} from desc2 
                is {comparison["multiplier"]} {x} <larger/smaller/unchanged> than the baseline of {val1} from desc1.
                The {metric2} of {val2} from desc2
                is an {y} <increase/decrease> of {comparison["percent_of_y"]} compared to the baseline of {val1} from desc1.
    
                The {metric2} changed by a factor of {comparison["percent_change_from_y"]} compared to the baseline of {val1} from desc1."""
            descriptions.append(" ".join(text.split()))
        elif(x =="unchanged"):
            text = f"""
                For
                desc1 for the {colname1} the {metric1} is {val1}
                compared to
                desc2 for the {colname2} the {metric2} is {val2}.

                Comparing both, 
                the {metric2} of {val2} from desc2 is unchanged compared to the baseline of {val1} from desc1."""
            descriptions.append(" ".join(text.split()))

        else:
            print("Error")
            descriptions.append("Error")
    final_df[colname1] = descriptions
    
print("final df: ", final_df.head())
final_df.to_json("helpers/jsons/final_df.jsonl", orient="columns", indent=0) # indent = 0 for inference!



