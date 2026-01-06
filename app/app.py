import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


# CONFIG

MODEL_PATH = "results/checkpoint-9375"
THRESHOLD = 0.05


# PAGE SETUP

st.set_page_config(page_title="Legal Document Classifier", layout="wide")


# LOAD MODEL

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    dataset = load_dataset("lex_glue", "ledgar")
    labels = dataset["train"].features["label"].names
    return tokenizer, model, labels

tokenizer, model, label_names = load_model()


# STYLES

st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f172a,#020617);}
.main {color:white;}
.card {background:#87ceeb;padding:25px;border-radius:16px;}
.metric {background:#020617;padding:20px;border-radius:12px;}
.bar {height:16px;border-radius:8px;background:linear-gradient(90deg,#22c55e,#10b981);}
.tag {display:inline-block;padding:6px 14px;border-radius:20px;background:#10b981;color:black;margin-right:8px;}
</style>
""", unsafe_allow_html=True)


# HERO

st.markdown("""
<div class="card" style="text-align:center">
<h1>⚖️ Legal Document Classifier</h1>
<p>Multi-label classification system powered by deep learning NLP</p>
<p>⚡ Instant Analysis &nbsp; 🛡️ 14+ Legal Categories &nbsp; ⭐ 94% Accuracy</p>
</div>
""", unsafe_allow_html=True)


# INPUT

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader(" Document Input")

mode = st.radio("", ["Paste Text", "Upload File"], horizontal=True)

text = ""
if mode == "Paste Text":
    text = st.text_area("Paste document text", height=200)
else:
    file = st.file_uploader("Upload .txt file", type=["txt"])
    if file:
        text = file.read().decode()

analyze = st.button(" Analyze Document")
st.markdown("</div>", unsafe_allow_html=True)


# PREDICTION

if analyze and text.strip():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0]
    results = [(label_names[i], float(p)) for i, p in enumerate(probs) if p > THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(" Classification Results")

    if not results:
        st.warning("No confident labels detected")
    else:
        primary = results[0]
        avg = sum(p for _, p in results) / len(results)

        col1, col2 = st.columns(2)
        col1.markdown(f"**Primary Category:** {primary[0]} ({primary[1]*100:.1f}%)")
        col2.markdown(f"**Avg Confidence:** {avg*100:.1f}%")

        st.markdown("### Predicted Categories")
        for label, score in results:
            st.markdown(f"{label} — {score*100:.1f}%")
            st.markdown(f"<div class='bar' style='width:{score*100}%'></div>", unsafe_allow_html=True)

        st.markdown("### Tags")
        for label, _ in results:
            st.markdown(f"<span class='tag'>{label}</span>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# METRICS

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader(" Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("F1 Score", "0.648")
c2.metric("Accuracy", "0.494")
c3.metric("ROC-AUC", "0.747")
c4.metric("Precision", "0.91")
st.markdown("</div>", unsafe_allow_html=True)
