from src.pipeline.analyze_comment import analyze_comment

comments = [
    "Nice pic bro 🔥",
    "Tu pagal hai 😂",
    "Tu gandu hai 🤬🤬"
]

for c in comments:
    score, action = analyze_comment(c)
    print(c)
    print(f"Score: {score:.2f} | Action: {action}")
    print("-" * 40)
