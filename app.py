import streamlit as st
import json

st.set_page_config(page_title="SRH", page_icon="🎙️", layout="wide")

st.title("SeqHub Research Hackathon — Financial Call Intelligence Pipeline")
st.caption("Reach Hackathon 2026 | Comfort Adeosun")

st.divider()

# Metrics
st.subheader(" Evaluation Results")
with open("outputs/metrics.json") as f:
    metrics = json.load(f)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("WER", metrics["WER"], delta="lower is better", delta_color="inverse")
col2.metric("Macro-F1", metrics["Macro_F1"], delta="higher is better")
col3.metric("NMI", metrics["NMI"], delta="higher is better")
col4.metric("C_v", metrics["C_v"], delta="higher is better")
col5.metric("WindowDiff", metrics["WindowDiff"], delta="lower is better", delta_color="inverse")
col6.metric("Pk", metrics["Pk"], delta="lower is better", delta_color="inverse")

st.divider()

# Speaker turns
st.subheader(" Speaker Turns — Sentiment & Topics")

with open("outputs/aligned_output.json") as f:
    aligned = json.load(f)

sentiment_colors = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral": "🔵",
    "mixed": "🟡"
}

for a in aligned:
    emoji = sentiment_colors.get(a["pred_sentiment"], "⚪")
    st.markdown(f"{emoji} **{a['gt_speaker']}** | `{a['gt_topic']}` | Sentiment: **{a['pred_sentiment']}**")
    st.write(a["text"])
    st.divider()
