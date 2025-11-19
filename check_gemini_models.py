import google.generativeai as genai

# 🔹 Configure your Gemini API key
genai.configure(api_key="AIzaSyC_VSwkpB_Os5_FsjF8eDuH4pZJAy-Io2w")  # or st.secrets["GEMINI_API_KEY"]

# 🔹 List all available models
models = genai.list_models()

for m in models:
    print(f"🧩 Model: {m.name}")
    print(f"   • Supported methods: {m.supported_generation_methods}")
    print("--------------------------------------------------")
