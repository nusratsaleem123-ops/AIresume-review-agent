# 📄 AI Resume Review Agent (CrewAI + Streamlit + Groq)

An intelligent, beginner-friendly single-agent resume evaluation application built using **CrewAI**, **Streamlit**, and **Groq**. 

This application takes a candidate's resume (either pasted raw text or an uploaded PDF file) along with a targeted job description, analyzes candidate alignment using an expert AI Recruiter agent, and generates structured, actionable recommendations without hallucinating unmentioned qualifications.

---

## 📁 Repository File Structure

```text
resume-review-agent/
│
├── .streamlit/
│   └── secrets.toml          # Local secret key storage (DO NOT push to GitHub)
├── app.py                    # Main Streamlit application code
├── requirements.txt          # Python library dependencies
└── README.md                 # Project documentation and setup guide