import pandas as pd
from supabase import create_client
import subprocess, json

# --- Setup ---
SUPABASE_URL = "https://havlidcifwdpmpswnsqr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhhdmxpZGNpZndkcG1wc3duc3FyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2Mjg2ODg1NSwiZXhwIjoyMDc4NDQ0ODU1fQ.uOulhzJxKNP1tpZUjUq9YGRFa0DGD0NWpbAlc7izpLg"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------
#   LOCAL EMBEDDING FUNCTION — (CORRECT VERSION)
# ---------------------------------------------------
def embed_local(text: str):
    """
    Generate embeddings using nomic-embed-text with stdin piping.
    Works on Windows + Ollama.
    """
    proc = subprocess.run(
        ["ollama", "run", "nomic-embed-text"],
        input=text.encode("utf-8"),
        capture_output=True
    )

    output = proc.stdout.decode("utf-8").strip()

    if not output:
        print("Ollama returned EMPTY output. Model may not support embeddings.")
        raise ValueError("Empty output from Ollama")

    try:
        data = json.loads(output)

        # If output is {"embedding": [...]}
        if isinstance(data, dict) and "embedding" in data:
            return data["embedding"]

        # If output is a list
        if isinstance(data, list):
            return data

        raise ValueError("Unexpected embedding structure")

    except Exception as e:
        print("RAW OUTPUT:", output)
        raise ValueError("Could not parse Ollama embedding output") from e


# ---------------------------------------------------
#   FETCH ROWS
# ---------------------------------------------------
resp = supabase.table("fuel_sales_transactions").select("*").execute()
df = pd.DataFrame(resp.data)

# ---------------------------------------------------
#   SUMMARY TEXT FOR EACH ROW
# ---------------------------------------------------
def make_summary(row):
    return (
        f"Truck {row['vehicle_number']} purchased {row['quantity_litres']} litres "
        f"for ₹{row['total_amount_rs']} at {row['station_name']} on {row['transaction_date']}. "
        f"Route {row['route_km']} km with avg mileage {row['avg_mileage']}."
    )

# ---------------------------------------------------
#   GENERATE EMBEDDINGS & SAVE
# ---------------------------------------------------
for idx, row in df.iterrows():
    text = make_summary(row)
    embedding = embed_local(text)

    supabase.table("fuel_sales_transactions").update({
        "text_to_embed": text,
        "embedding": embedding
    }).eq("id", row["id"]).execute()

print("🔥 Done — embeddings saved using local model!")
