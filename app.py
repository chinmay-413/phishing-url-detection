import json, os
import numpy as np, pandas as pd, altair as alt, streamlit as st
from predictor import Detector, ART

st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")

COLORS = {"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"}
ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}
MSG = {"Low": "No strong phishing signals found.",
       "Medium": "Some suspicious signals. Be careful and do not enter passwords.",
       "High": "Looks like phishing. Do not open it or enter any information."}
EXAMPLES = {"✅ Wikipedia": "https://www.wikipedia.org/",
            "✅ GitHub": "https://github.com/",
            "⚠️ Fake PayPal": "http://paypa1-secure-login.com/verify?account=123",
            "⚠️ Fake bank": "http://secure-sbi-update.verify-account.xyz/login.php"}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {padding: 1.2rem 1.5rem; border-radius: 16px; margin-bottom: 1rem;
       background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%); color: white;}
.hero h1 {margin: 0; font-size: 2rem; color: white;} .hero p {margin: .3rem 0 0; opacity: .92;}
.result {display: flex; gap: 1.2rem; align-items: center; padding: 1rem 1.3rem; border-radius: 14px;
         background: rgba(128,128,128,.10);}
.big {font-size: 3.2rem; font-weight: 800; line-height: 1;}
.lvl {font-size: 1.2rem; font-weight: 700;} .sub {opacity: .8;}
.bar {height: 10px; border-radius: 6px; background: rgba(128,128,128,.25); margin: .6rem 0 1rem; overflow: hidden;}
.bar div {height: 100%; border-radius: 6px;}
</style>
<div class="hero"><h1>🛡️ PhishGuard AI</h1>
<p>Phishing URL detection with an ensemble of three models trained on two datasets, with explanations.</p></div>
""", unsafe_allow_html=True)

if not os.path.exists(os.path.join(ART, "bundle.joblib")):
    st.error("No trained model found. Run `python train.py` first, then reload this page.")
    st.stop()


@st.cache_resource
def load_detector():
    return Detector()


@st.cache_data
def load_artifacts():
    p = lambda n: os.path.join(ART, n)
    out = {"summary": json.load(open(p("summary.json"))), "curves": json.load(open(p("curves.json")))}
    for k, f in {"metrics": "metrics.csv", "importance": "importance.csv", "sample": "analysis_sample.csv", "errors": "errors.csv"}.items():
        out[k] = pd.read_csv(p(f))
    return out


det = load_detector()
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("How to read the score")
    st.markdown("🟢 **Low**: no strong signals  \n🟠 **Medium**: suspicious, be careful  \n🔴 **High**: likely phishing")
    st.caption(f"High-risk cut-off (tuned on validation data): {det.thr:.0%}")
    st.divider()
    st.caption("This is a decision-support tool, not a guarantee. When in doubt, type the website address yourself instead of clicking a link.")

tab_scan, tab_batch, tab_insights, tab_about = st.tabs(["🔍 Scan a URL", "📦 Batch scan", "📊 Model insights", "ℹ️ About"])

# ------------------------------------------------------------------ scan
with tab_scan:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for c, (label, u) in zip(cols, EXAMPLES.items()):
        c.button(label, use_container_width=True, on_click=lambda u=u: st.session_state.update(url=u))
    url = st.text_input("Paste a URL and press Enter", key="url", placeholder="https://example.com/login")

    if url.strip():
        r = det.analyze(url.strip())
        if not st.session_state.history or st.session_state.history[-1]["url"] != url.strip():
            st.session_state.history.append({"url": url.strip(), "risk": round(r["risk"], 3), "level": r["level"]})
        col = COLORS[r["level"]]
        st.markdown(f"""<div class="result" style="border-left:8px solid {col}">
            <div class="big" style="color:{col}">{r['risk']:.0%}</div>
            <div><div class="lvl" style="color:{col}">{r['level'].upper()} RISK</div>
            <div class="sub">{MSG[r['level']]}</div></div></div>
            <div class="bar"><div style="width:{r['risk']*100:.0f}%;background:{col}"></div></div>""",
                    unsafe_allow_html=True)
        if r["known_domain"]:
            st.success("This is a well-known official domain, so the risk is capped at a low value.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Feature model", f"{r['feature_model']:.0%}", help="Gradient boosting on 30 URL features")
        m2.metric("URL-text model", f"{r['text_model']:.0%}", help="Character n-grams of the URL text")
        m3.metric("Pooled-dataset model", f"{r['pooled_model']:.0%}", help="Random forest trained on both datasets together")

        left, right = st.columns(2)
        with left:
            st.subheader("Warning signs")
            if r["flags"]:
                for sev, text in r["flags"]:
                    st.markdown(f"{ICON[sev]} {text}")
            else:
                st.markdown("✅ No common red flags found.")
        with right:
            st.subheader("What pushed the score")
            c = r["contrib"].head(8)
            if len(c):
                d = pd.DataFrame({"feature": c.index, "effect": c.values, "direction": np.where(c.values > 0, "raises risk", "lowers risk")})
                st.altair_chart(alt.Chart(d).mark_bar().encode(
                    x=alt.X("effect:Q", title="Change in feature-model risk"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"], range=["#dc2626", "#16a34a"]), legend=None),
                    tooltip=["feature", alt.Tooltip("effect:Q", format="+.3f")]).properties(height=260), use_container_width=True)
                st.caption("Computed by replacing each feature with a typical safe value and measuring the change.")
            else:
                st.write("No single feature stands out.")
        with st.expander("All extracted features"):
            st.dataframe(r["features"].rename("value").to_frame(), use_container_width=True)

    if st.session_state.history:
        st.subheader("Scan history (this session)")
        h = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(h, use_container_width=True, hide_index=True)
        st.download_button("Download history (CSV)", h.to_csv(index=False), "scan_history.csv", "text/csv")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Check many URLs at once: paste them (one per line) or upload a CSV that has a `url` column. Maximum 3000.")
    txt = st.text_area("URLs", height=140, placeholder="https://example.com\nhttp://suspicious-login.xyz")
    up = st.file_uploader("...or upload a CSV", type="csv")
    if st.button("Scan all", type="primary"):
        urls = [u for u in txt.splitlines() if u.strip()]
        if up is not None:
            f = pd.read_csv(up)
            col_name = next((c for c in f.columns if c.lower() == "url"), f.columns[0])
            urls += f[col_name].dropna().astype(str).tolist()
        if not urls:
            st.warning("Add some URLs first.")
        else:
            if len(urls) > 3000:
                st.warning("Only the first 3000 URLs are scanned.")
                urls = urls[:3000]
            with st.spinner(f"Scanning {len(urls)} URLs ..."):
                res = det.score_many(urls)
            st.session_state.batch = res

    if "batch" in st.session_state:
        res = st.session_state.batch
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Scanned", len(res))
        for k, lv in zip((k2, k3, k4), ("High", "Medium", "Low")):
            k.metric(f"{lv} risk", int((res.level == lv).sum()))
        color = lambda v: f"background-color: {COLORS[v]}33; font-weight: 600"
        st.dataframe(res.sort_values("risk", ascending=False).style.map(color, subset=["level"]).format(
            {"risk": "{:.0%}", "feature_model": "{:.0%}", "text_model": "{:.0%}", "pooled_model": "{:.0%}"}),
            use_container_width=True, hide_index=True)
        st.download_button("Download results (CSV)", res.to_csv(index=False), "batch_results.csv", "text/csv")

# ------------------------------------------------------------------ insights
with tab_insights:
    try:
        art = load_artifacts()
    except Exception as e:
        st.warning(f"Analysis files not found ({e}). Run `python train.py` again.")
        st.stop()
    s, mt = art["summary"], art["metrics"]
    st.caption("All scores below are measured on websites the models never saw during training (domain-separated test set).")
    a, b, c, d = st.columns(4)
    a.metric("URLs used (after augmentation)", f"{s['rows_after_augmentation']:,}")
    b.metric("Test URLs", f"{s['test']:,}")
    c.metric("Domains shared train/test", s["shared_domains"])
    d.metric("Phishing share", f"{s['phishing_share']:.0%}")

    st.subheader("Model comparison")
    num = [c for c in mt.columns if c != "Model"]
    st.dataframe(mt.style.highlight_max(subset=num, color="#16a34a55").format({c: "{:.3f}" for c in num}),
                 use_container_width=True, hide_index=True)
    long = mt.melt("Model", ["Precision", "Recall", "F1-score"], var_name="metric", value_name="score")
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30)), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
        xOffset="metric:N", color="metric:N", tooltip=["Model", "metric", alt.Tooltip("score:Q", format=".3f")]).properties(height=320),
        use_container_width=True)

    g1, g2 = st.columns(2)
    for col, key, title, xt, yt in ((g1, "roc", "ROC curve", "False positive rate", "True positive rate"),
                                    (g2, "pr", "Precision-recall curve", "Recall", "Precision")):
        cv = pd.concat([pd.DataFrame({"x": v["x"], "y": v["y"], "model": n}) for n, v in art["curves"][key].items()])
        col.subheader(title)
        col.altair_chart(alt.Chart(cv).mark_line().encode(x=alt.X("x:Q", title=xt), y=alt.Y("y:Q", title=yt), color="model:N",
                         strokeWidth=alt.condition(alt.datum.model == "ENSEMBLE (final)", alt.value(4), alt.value(1.5))).properties(height=300),
                         use_container_width=True)

    h1, h2 = st.columns(2)
    cm = np.array(s["confusion"])
    cmd = pd.DataFrame([{"actual": ["Legitimate", "Phishing"][i], "predicted": ["Legitimate", "Phishing"][j], "count": int(cm[i, j])}
                        for i in range(2) for j in range(2)])
    base = alt.Chart(cmd).encode(x="predicted:N", y="actual:N")
    h1.subheader("Confusion matrix (ensemble)")
    h1.altair_chart((base.mark_rect().encode(color=alt.Color("count:Q", legend=None, scale=alt.Scale(scheme="blues"))) +
                     base.mark_text(fontSize=18).encode(text="count:Q")).properties(height=280), use_container_width=True)
    h2.subheader("Most important features")
    imp = art["importance"].head(12)
    h2.altair_chart(alt.Chart(imp).mark_bar().encode(x=alt.X("importance:Q", title="Drop in score when shuffled"),
                    y=alt.Y("feature:N", sort="-x", title=None)).properties(height=280), use_container_width=True)

    st.subheader("Dataset analysis")
    smp = art["sample"].copy()
    smp["class"] = np.where(smp["label"] == 1, "Phishing", "Legitimate")
    feat = st.selectbox("Compare a feature between the classes", [c for c in smp.columns if c not in ("label", "class")], index=0)
    st.altair_chart(alt.Chart(smp).mark_bar(opacity=0.6).encode(
        x=alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=30)), y=alt.Y("count()", stack=None, title="URLs"),
        color=alt.Color("class:N", scale=alt.Scale(domain=["Legitimate", "Phishing"], range=["#16a34a", "#dc2626"]))).properties(height=280),
        use_container_width=True)

    st.subheader("Dataset 2 (pooled model on its own hold-out)")
    st.json(s["pooled_on_d2_holdout"])
    st.caption(f"Dataset 2 label check AUC = {s['d2_label_check_auc']} (target flipped: {s['d2_target_flipped']}).")

    st.subheader("Error analysis")
    st.write("Test URLs the ensemble got wrong. Looking at these shows what the model finds hard.")
    st.dataframe(art["errors"][["kind", "url", "risk"]].style.format({"risk": "{:.0%}"}), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
### How it works
1. **Feature model:** gradient boosting on about 30 URL features (length, digits, entropy, subdomains, brand look-alikes, risky endings, and so on).
2. **URL-text model:** logistic regression on character n-grams, which learns suspicious spellings and patterns.
3. **Pooled model:** a random forest trained on the shared numeric features of **both datasets together**.
4. A small **stacking model** combines the three scores. Well-known official domains are capped at low risk.

### What makes it more reliable
- Test sites are never seen in training (domain-separated split).
- Training includes bare-domain and root-page versions of each URL, which removes a dataset shortcut ("short URL = phishing").
- Every result comes with warning signs and a what-if explanation.

### Limitations
- Only the URL text is analysed: no page content, domain age, certificates or live blocklists.
- Datasets age quickly, so the model must be retrained as attacks change.
- New or unusual legitimate sites can be flagged, and some phishing will be missed.
""")