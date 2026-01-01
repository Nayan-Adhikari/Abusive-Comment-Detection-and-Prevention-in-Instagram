from src.pipeline.analyze_comment import analyze_comment

if __name__ == "__main__":
    print("Instagram Toxic Comment Moderation (STC)")
    print("-" * 50)

    while True:
        comment = input("\nEnter a comment (or type 'exit'): ")
        if comment.lower() == "exit":
            break

        score, action = analyze_comment(comment)

        print(f"Toxicity Score : {score:.2f}")
        print(f"Action         : {action}")
