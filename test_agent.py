from rag_agent import build_chain

chain = build_chain()
profile = "Name: Test\nEducation: B.Tech\nSkills: Python, SQL\nInterests: data and AI"

tests = [
    "Which career suits me?",
    "Give me a roadmap to become a Data Scientist.",
    "What certifications help for cybersecurity?",
    "What is the salary of an astronaut?",   # out-of-context check
]

for q in tests:
    print(f"\nQ: {q}")
    print("A:", chain.invoke({"question": q, "profile": profile, "history": ""}))