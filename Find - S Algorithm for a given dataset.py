import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("enjoyablesport.csv")

# Extract features and target
attributes = df.columns[:-1]
target = df.columns[-1]

# FIND-S Algorithm
def find_s(examples):
    specific_hypothesis = ["?"] * len(attributes)
    
    for i, row in examples.iterrows():
        if row[target] == "yes":
            if specific_hypothesis == ["?"] * len(attributes):
                specific_hypothesis = row.iloc[:-1].tolist()
            else:
                for j in range(len(specific_hypothesis)):
                    if specific_hypothesis[j] != row.iloc[j]:
                        specific_hypothesis[j] = "?"
    return specific_hypothesis

# Candidate-Elimination Algorithm
def candidate_elimination(examples):
    S = ["?"] * len(attributes)  # Most specific hypothesis
    G = [["?"] * len(attributes)]  # Most general hypothesis
    
    for _, row in examples.iterrows():
        if row[target] == "yes":
            for i in range(len(S)):
                if S[i] == "?":
                    S[i] = row.iloc[i]
                elif S[i] != row.iloc[i]:
                    S[i] = "?"
            G = [g for g in G if all(g[i] == "?" or g[i] == S[i] for i in range(len(S)))]
        else:
            G = [g.copy() for g in G]
            for i in range(len(attributes)):
                if S[i] != "?":
                    new_hypothesis = S.copy()
                    new_hypothesis[i] = "?"
                    if new_hypothesis not in G:
                        G.append(new_hypothesis)
    return S, G

# Run algorithms
specific_hypothesis = find_s(df)
S_final, G_final = candidate_elimination(df)

print("Most Specific Hypothesis (Find-S):", specific_hypothesis)
print("Final Specific Hypothesis (Candidate-Elimination):", S_final)
print("Final General Hypothesis (Candidate-Elimination):", G_final)
