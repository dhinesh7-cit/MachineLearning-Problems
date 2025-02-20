import pandas as pd

# Initialize the specific and general boundaries
def initialize_boundaries(attributes):
    specific = ['0'] * len(attributes)
    general = [['?'] * len(attributes)]
    return specific, general

# Check if a hypothesis is consistent with an example
def is_consistent(hypothesis, example):
    for i in range(len(hypothesis)):
        if hypothesis[i] != '?' and hypothesis[i] != example[i]:
            return False
    return True

# Update the specific boundary
def update_specific(specific, example):
    for i in range(len(specific)):
        if specific[i] == '0':
            specific[i] = example[i]
        elif specific[i] != example[i]:
            specific[i] = '?'
    return specific

# Update the general boundary
def update_general(general, example, specific):
    new_general = []
    for hypo in general:
        if is_consistent(hypo, example):
            new_general.append(hypo)
        else:
            for i in range(len(hypo)):
                if hypo[i] != '?' and hypo[i] != specific[i]:
                    new_hypo = hypo.copy()
                    new_hypo[i] = '?'
                    if is_consistent(new_hypo, specific):
                        new_general.append(new_hypo)
    return new_general

# Candidate elimination algorithm
def candidate_elimination(data, attributes):
    specific, general = initialize_boundaries(attributes)
    
    for index, row in data.iterrows():
        example = list(row.iloc[:-1])  # Use iloc to access by position
        label = row.iloc[-1]  # Use iloc to access by position
        
        if label == 'Yes':  # Positive example
            specific = update_specific(specific, example)
            general = update_general(general, example, specific)
        else:  # Negative example
            general = [hypo for hypo in general if not is_consistent(hypo, example)]
    
    return specific, general

# Dataset
data = pd.DataFrame({
    'Type': ['Ball', 'Racket', 'Bat', 'Ball', 'Basketball', 'Golf Ball', 'Book', 'Frisbee', 'Shoe', 'Tennis Ball'],
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Large', 'Small', 'Large', 'Medium', 'Medium', 'Small'],
    'Weight': ['Light', 'Light', 'Medium', 'Heavy', 'Heavy', 'Light', 'Heavy', 'Light', 'Heavy', 'Light'],
    'Color': ['Red', 'Blue', 'Black', 'Yellow', 'Orange', 'White', 'Brown', 'Green', 'Black', 'Yellow'],
    'Playable': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
})

# Attributes
attributes = ['Type', 'Size', 'Weight', 'Color']

# Run the algorithm
specific, general = candidate_elimination(data, attributes)

# Output the results
print("Specific Boundary (S):", specific)
print("General Boundary (G):", general)
for h in general:
    print (h)
