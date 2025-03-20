# Code to update the test.json files with category type based on question length
''' Remove the comment to use the below code
import json

# Load test.json
test_json_path = "dataset/test_1000.json"
updated_json_path = "dataset/test_1000_updated.json"  # Save the modified JSON

def categorize_question(question):
    """Categorize the question based on word count."""
    word_count = len(question.split())
    if word_count <= 5:
        return "Short"
    elif word_count <= 10:
        return "Medium"
    else:
        return "Long"

# Load JSON data
with open(test_json_path, "r") as file:
    test_data = json.load(file)

# Add category field to each question
for item in test_data:
    item["category"] = categorize_question(item["question"])

# Save the updated JSON
with open(updated_json_path, "w") as file:
    json.dump(test_data, file, indent=2)

print(f"Updated JSON saved as {updated_json_path}")

End of the comment''' 

# Code to calculate the average AudoSight Resposne time based on the Question Category

import re
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import pandas as pd
'''

def extract_and_average_response_times(file_path):
    # Dictionary to store response times by category
    response_times = defaultdict(list)
    
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Find all occurrences of categories and their AI response times
    matches = re.findall(r'Category:\s*(\w+).*?AI Response Time:\s*([\d\.]+)', content, re.DOTALL)
    
    # Populate dictionary with extracted values
    for category, time in matches:
        response_times[category].append(float(time))
    
    # Compute averages
    average_response_times = {cat: sum(times) / len(times) for cat, times in response_times.items()}
    
    return average_response_times

def plot_response_times(average_response_times):
    categories = list(average_response_times.keys())
    times = list(average_response_times.values())
    
    plt.figure(figsize=(8, 5))
    plt.bar(categories, times, color=['lightblue', 'lightgreen', 'tomato'])

    plt.xlabel("Response Category")
    plt.ylabel("Average AudoSight Response Time (seconds)")
    plt.title("Average AudoSight Response Time by Category")
    plt.show()

file_path = "results/results.txt"  
averages = extract_and_average_response_times(file_path)

print(averages)
plot_response_times(averages)
'''
# Code for Latency Distribution for 100 different Query Lenghts 
import re
'''
def parse_results(file_path):
    results = {}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # Split by image entries
        entries = content.strip().split('\n\n')
        
        for entry in entries:
            category_match = re.search(r'Category: (.+)', entry)
            total_time_match = re.search(r'AI Response Time: ([\d\.]+) seconds', entry)
            
            if category_match and total_time_match:
                category = category_match.group(1)
                total_time = float(total_time_match.group(1))
                
                if category not in results:
                    results[category] = []
                results[category].append(total_time)
    
    return results

file_path = 'results/results.txt'  
parsed_data = parse_results(file_path)

data = {
    "Query Length": ["Short"] * 100 + ["Medium"] * 100 + ["Long"] * 100,
    "Latency (ms)": [*parsed_data['Short'][:100],   # Short
                    *parsed_data['Medium'][:100],# Medium
                   *parsed_data['Long'][:100]]  # Long
}

df = pd.DataFrame(data)

# Boxplot for latency distribution across query lengths
plt.figure(figsize=(8, 6))
sns.boxplot(x="Query Length", y="Latency (ms)", data=df, palette="Blues")
plt.title("Latency Distribution for Different Query Lengths")
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt

# X-axis labels (Without and With Guardrails)
iterations = ["Without NemoGuardrails", "With NemoGuardrails"]
Total_query_count=100
blocked_queries_count_type1=50
blocked_queries_count_type2=50
blocked_queries_count_type3=50
percentage_type1=(blocked_queries_count_type1/Total_query_count)*100
percentage_type1=(blocked_queries_count_type2/Total_query_count)*100
percentage_type1=(blocked_queries_count_type3/Total_query_count)*100
# Percentage of blocked queries for each unsafe category (Y-axis)
blocked_queries = {
    "Inaccessible Instructions": [11, 0],
    "Admiring Beauty in a Way that Evokes Pity": [10, 0],
    "Vague Responses": [9, 0],
}

# Plot the data
plt.figure(figsize=(8, 5))
for category, values in blocked_queries.items():
    plt.plot(iterations, values, marker='o', label=category)

# Graph labels and title
plt.xlabel("Guardrails Implementation")
plt.ylabel("Percentage of Inappropriate responses (%)")
plt.title("Inappropirate Responses Reduction with Guardrails")
plt.legend(title="Unsafe Categories")
plt.grid(True)

# Show the graph
plt.show()
