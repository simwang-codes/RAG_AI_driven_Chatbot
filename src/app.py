import streamlit as st
import sys
sys.modules["torch.classes"] = None  # Prevent Streamlit from inspecting torch.classes
from rag import search_similar_chunks, answer_question_with_prompt  # ← updated import

def run_app():
    st.set_page_config(page_title="📚 RAG for Video Game History", layout="wide")

    st.title("👾 Video Game History Q&A 👾")
    st.markdown("Please ask a question related to video game history, I will answer your question! Feel free to use any languages you like!")

    query = st.text_input("💬 Type Your Question Here：", placeholder="For instance, Why Nintendo succeeded in American market？")

    if st.button("🔍 Query") and query.strip():
        with st.spinner("Searching for an answer..."):
            top_docs = search_similar_chunks(query, k=5)
            answer = answer_question_with_prompt(query, top_docs)

            st.subheader("💬 Answer：")
            st.markdown(answer)

            st.divider()
            st.subheader("🔎 Matched Content Chunks")
            for i, doc in enumerate(top_docs):
                st.markdown(f"**📄 Chunk {i+1}**")
                st.markdown(f"- **Filename**：{doc.metadata.get('filename', '')}")
                st.markdown(f"- **Tags**：{doc.metadata.get('tags', '')}")
                st.code(doc.page_content.strip()[:1000], language="markdown")

# Run the app
if __name__ == "__main__":
    run_app()
