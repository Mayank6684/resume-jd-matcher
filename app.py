import os
import json
from dotenv import load_dotenv
import streamlit as st
import pdfplumber
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# --- Setup LLM ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical recruiter comparing a resume to a job description."),
    ("user", """Compare this resume and job description. Respond ONLY with valid JSON, no other text, in this exact format:
{{
  "match_score": <number 0-100>,
  "matched_skills": [...],
  "missing_skills": [...],
  "suggestions": [...]
}}

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
""")
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# --- Helper: extract text from PDF ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Helper: clean and parse JSON from LLM response ---
def parse_llm_json(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    return json.loads(cleaned)

# --- Streamlit UI ---
st.set_page_config(page_title="Resume ↔ JD Matcher", page_icon="📄")
st.title("Resume ↔ JD Matcher")
st.caption("Upload your resume and paste a job description to see how well they match.")

resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
jd_input = st.text_area("Paste job description", height=200)

if st.button("Analyze", type="primary"):
    if not resume_file or not jd_input.strip():
        st.warning("Please upload a resume and paste a job description first.")
    else:
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(resume_file)

            try:
                raw_response = chain.invoke({"resume": resume_text, "jd": jd_input})
                result = parse_llm_json(raw_response)

                st.metric("Match Score", f"{result['match_score']}/100")

                st.subheader("✅ Matched Skills")
                st.write(", ".join(result["matched_skills"]) if result["matched_skills"] else "None found")

                st.subheader("❌ Missing Skills")
                st.write(", ".join(result["missing_skills"]) if result["missing_skills"] else "None")

                st.subheader("💡 Suggestions")
                for s in result["suggestions"]:
                    st.write(f"- {s}")

            except json.JSONDecodeError:
                st.error("Couldn't parse the AI's response. Raw output below:")
                st.text(raw_response)
            except Exception as e:
                st.error(f"Something went wrong: {e}")