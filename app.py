import streamlit as st
from transformers import pipeline
import pdfplumber
from fpdf import FPDF
import io
import datetime

# Page config & CSS
st.set_page_config(page_title="AI Notes Summarizer", page_icon="📝", layout="centered")
st.markdown("""
<style>
.stButton > button {
  background-color: #d9d4fc;   /* soft pastel purple */
  color: #4b4b4b;
  border-radius: 10px;
  padding: 8px 16px;
  font-size: 16px;
  border: none;
}
</style>

""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>📝 AI Notes Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload PDFs or paste notes — get clean summaries instantly.</p>", unsafe_allow_html=True)

# Load model once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

with st.spinner("Loading AI model..."):
    summarizer = load_summarizer()
 
# PDF Upload & Text extraction
st.write("### 📂 Upload a PDF file (optional):")
pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

extracted_text = ""
if pdf_file:
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() or ""
        if not extracted_text.strip():
            st.warning("PDF uploaded but no text was extracted.")
        else:
            st.success("PDF text extracted successfully!")
            st.write("#### 📄 Extracted Text Preview:")
            st.text_area("", extracted_text[:1500] + ("..." if len(extracted_text) > 1500 else ""), height=180)
    except Exception as e:
        st.error(f"Failed to extract PDF text: {e}")

# Manual text area
st.write("### ✍️ Or paste text manually:")
user_text = st.text_area("", height=250, placeholder="Enter long notes, textbooks, articles...")

final_text = (user_text.strip() + "\n\n" + extracted_text.strip()).strip() if (user_text.strip() or extracted_text.strip()) else ""

# Word / Character count
if final_text:
    words = final_text.split()
    char_count = len(final_text)
    word_count = len(words)
    st.write(f"**Words:** {word_count}  •  **Characters:** {char_count}")

# Summary options

length_option = st.selectbox(" Select summary length:", ["Short", "Medium", "Long"])
length_map = {"Short": (40, 120), "Medium": (80, 200), "Long": (120, 300)}
min_len, max_len = length_map[length_option]

# Buttons: Generate, Clear
col1, col2 = st.columns(2)

with col1:
    generate = st.button("✨ Generate Summary")

with col2:
    clear = st.button(" Clear / Reset")

summary_text = ""
if generate:
    if len(final_text) < 50:
        st.warning("Please enter at least 50 characters or upload a PDF with text.")
    else:
        try:
            with st.spinner("Summarizing..."):
                res = summarizer(final_text, max_length=max_len, min_length=min_len, do_sample=False)
                summary_text = res[0]["summary_text"].strip()
            st.success("Summary generated!")
            st.write("### 🔍 Summary:")
            st.write(summary_text)
        except Exception as e:
            st.error(f"Error during summarization: {e}")

if clear:
    st.experimental_rerun()

# Download TXT + PDF
if summary_text:
    # TXT
    txt_bytes = summary_text.encode("utf-8")
    filename_txt = f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button(label="⬇️ Download as TXT", data=txt_bytes, file_name=filename_txt, mime="text/plain")

    # PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 8, txt=line)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    filename_pdf = f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    st.download_button(label="⬇️ Download as PDF", data=pdf_bytes, file_name=filename_pdf, mime="application/pdf")

st.markdown("</div>", unsafe_allow_html=True)
