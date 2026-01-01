from src.pipeline.analyze_comment import analyze_comment

def test_sample_comment():
    score, action = analyze_comment("Tu pagal hai 😂")
    assert 0.0 <= score <= 1.0
