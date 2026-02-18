import os
import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient
import transformers


MODEL_NAME = "nateraw/bert-base-uncased-emotion"


@st.cache_resource
def get_local_pipeline():
    try:
        pipe = transformers.pipeline(
            "text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            return_all_scores=True,
            device=-1,
        )
        return pipe
    except Exception:
        return None


@st.cache_resource
def get_hf_client():
    hf_token="hftoken"
    #token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None
    return InferenceClient(provider="hf-inference", api_key=hf_token)


def predict_with_client(client, text):
    res = client.text_classification(text, model=MODEL_NAME)
    # res is list of {label, score}
    return [{"label": r["label"], "score": r["score"]} for r in res]


def predict_with_pipeline(pipe, text):
    res = pipe(text)
    # pipeline returns list (batch) -> take first
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        return [{"label": r["label"], "score": r["score"]} for r in res[0]]
    # fallback
    return res


def main():
    st.title("Emotion detection — enter a sentence to classify")

    hf_client = get_hf_client()
    

    if hf_client:
        st.info("Using Hugging Face Inference API (HF_TOKEN found).")
    else:
        st.warning("No HF_TOKEN and local pipeline unavailable. Install transformers or set HF_TOKEN.")

    text = st.text_area("Enter a sentence", value="I feel great today!", height=120)

    if st.button("Predict"):
        if not text.strip():
            st.error("Please enter some text to classify.")
            return

        with st.spinner("Predicting..."):
            try:
                if hf_client:
                    out = predict_with_client(hf_client, text)
                else:
                    st.error("No prediction backend available.")
                    return

                df = pd.DataFrame(out)
                df = df.rename(columns={"label": "class", "score": "probability"})
                df = df.sort_values("probability", ascending=False).reset_index(drop=True)

                top = df.iloc[0]
                st.subheader(f"Predicted emotion: {top['class']} ({top['probability']:.2f})")
                st.table(df)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
