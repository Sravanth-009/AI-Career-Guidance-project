import streamlit as st
from rag_agent import build_chain, format_history

st.set_page_config(page_title="AI Career Guidance Agent", page_icon="🎯")
st.title("🎯 AI Career Guidance Agent")
st.caption("Get personalised career advice based on your skills and interests. Powered by Groq.")


@st.cache_resource
def get_chain():
    return build_chain()


chain = get_chain()

# Sidebar: student profile
with st.sidebar:
    st.header("Your Profile")
    name = st.text_input("Name")
    education = st.selectbox("Education", ["B.Tech", "B.Sc", "BCA", "B.Com", "Diploma", "Other"])
    skills = st.text_area("Your skills", placeholder="Python, SQL, communication...")
    interests = st.text_area("Your interests", placeholder="AI, design, security...")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

profile = f"Name: {name}\nEducation: {education}\nSkills: {skills}\nInterests: {interests}"

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about careers, skills, roadmaps..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = chain.invoke({
                    "question": question,
                    "profile": profile,
                    "history": format_history(st.session_state.messages[:-1]),
                })
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})