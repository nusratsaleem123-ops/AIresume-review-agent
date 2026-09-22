import streamlit as st
import pypdf
import time
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Single-Agent AI Resume Evaluator")
st.markdown("""
This application evaluates a candidate's resume against a targeted job description using a specialized **CrewAI Agent** powered by **Groq**.
""")

# -------------------------------------------------------------------
# Secret / Key Initialization
# -------------------------------------------------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY", None)

if not groq_api_key:
    st.error("⚠️ `GROQ_API_KEY` not found in Streamlit Secrets! Please configure `.streamlit/secrets.toml` locally or set it in Streamlit Cloud Settings.")
    st.stop()

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text from an uploaded PDF file safely."""
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n" + page_text
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error reading PDF file: {str(e)}")

# -------------------------------------------------------------------
# User Inputs
# -------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Candidate Resume")
    input_method = st.radio("Choose Input Method:", ["Upload PDF", "Paste Raw Text"])
    
    resume_text = ""
    if input_method == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])
        if uploaded_pdf:
            with st.spinner("Extracting text from PDF..."):
                try:
                    resume_text = extract_text_from_pdf(uploaded_pdf)
                    if resume_text:
                        st.success("PDF extracted successfully!")
                        with st.expander("Preview Extracted Resume"):
                            st.text(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))
                    else:
                        st.warning("The PDF appears to be empty or consists solely of images/scans without readable text.")
                except Exception as err:
                    st.error(f"Failed to process PDF: {err}")
    else:
        resume_text = st.text_area("Paste Candidate Resume Text:", height=250, placeholder="Paste complete resume content here...")

with col2:
    st.subheader("2. Target Job Description")
    job_description = st.text_area("Paste Job Description:", height=310, placeholder="Paste target job responsibilities and requirements here...")

# -------------------------------------------------------------------
# Core Agent Execution
# -------------------------------------------------------------------
st.markdown("---")
analyze_btn = st.button("🚀 Analyze Match & Generate Recommendations", type="primary")

if analyze_btn:
    # Validation checks
    if not resume_text.strip():
        st.warning("Please provide a resume (via PDF or text input) before proceeding.")
        st.stop()
        
    if not job_description.strip():
        st.warning("Please provide a Target Job Description before proceeding.")
        st.stop()

    with st.spinner("Agent analyzing match... Please wait."):
        try:
            # Initialize Groq LLM using ChatGroq wrapper
            llm = ChatGroq(
                temperature=0.2,
                groq_api_key=groq_api_key,
                model_name="openai/gpt-oss-120b"
            )

            # Define CrewAI Single Agent
            resume_evaluator = Agent(
                role="Senior Technical Recruiter and Resume Auditor",
                goal="Accurately measure candidate capability fit against job descriptions without fabricating details.",
                backstory="""You are an expert HR strategist and technical recruiter with 15+ years of experience.
                You are renowned for being strictly grounded: you NEVER fabricate or assume candidate experience, skills,
                or metrics that are not explicitly stated in the provided resume text. Your advice is objective, evidence-backed, and practical.""",
                verbose=False,
                allow_delegation=False,
                llm=llm
            )

            # Define Analysis Task
            evaluation_task = Task(
                description=f"""
                Analyze the candidate's resume against the target job description provided below.

                === RESUME TEXT ===
                {resume_text}

                === JOB DESCRIPTION ===
                {job_description}

                INSTRUCTIONS & GROUNDING RULES:
                1. Only use explicitly stated facts from the resume. DO NOT invent, assume, or hallucinate qualifications or candidate experience.
                2. Identify matched qualifications, critical missing skills/gaps, and alignment score.
                3. Provide clear, structured, actionable recommendations to improve resume alignment for this specific role.

                Format your response using clear Markdown headings:
                - ## Match Summary & Score (0-100%)
                - ## Strong Qualifications Met
                - ## Critical Skill Gaps & Missing Requirements
                - ## Actionable Resume Recommendations
                """,
                expected_output="A structured markdown report providing alignment evaluation and grounded improvement recommendations.",
                agent=resume_evaluator
            )

            # Assemble & Kickoff Crew
            crew = Crew(
                agents=[resume_evaluator],
                tasks=[evaluation_task],
                process=Process.sequential
            )

            # Execute Agent Run
            result = crew.kickoff()
            
            st.success("Analysis Complete!")
            st.markdown(result)

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                st.error("⏱️ Groq API Rate Limit reached! Please wait 30 seconds before trying again.")
            elif "401" in error_str or "invalid api key" in error_str.lower():
                st.error("🔑 Invalid Groq API Key! Please verify your key configuration.")
            else:
                st.error(f"An error occurred during evaluation: {error_str}")