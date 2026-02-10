import pandas as pd
import random

roles = ["General", "Philosopher", "Statesman", "Tyrant", "Architect"]
cities = ["Athens", "Sparta", "Corinth", "Thebes", "Rome"]
adjectives = ["brave", "cunning", "ruthless", "wise", "ancient"]
items = ["Sword", "Shield", "Scroll", "Statue", "Coin"]

data = []

print("Generating 500 historical items...")

# Generate 500 random items
for i in range(500):
    category = random.choice(["Person", "Location", "Artifact"])
    if category == "Person":
        name = f"Character_{i}"
        desc = f"A {random.choice(adjectives)} {random.choice(roles)} from {random.choice(cities)}."
    elif category == "Artifact":
        name = f"Artifact_{i}"
        desc = f"A {random.choice(adjectives)} {random.choice(items)} found in {random.choice(cities)}."
    else:
        name = f"Location_{i}"
        desc = f"The {random.choice(adjectives)} city of {random.choice(cities)}."
    
    data.append({"id": i, "name": name, "description": desc, "type": category})

df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
print("✅ Created 'data.csv' with 500 items!")