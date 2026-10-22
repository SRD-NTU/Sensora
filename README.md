# Sensora

## Run Online

Sensora is hosted on Streamlit Community Cloud: [https://sens0ra.streamlit.app](https://sensora-web.streamlit.app/)

You give it a pressure range, and it searches for the soft polymer matrix, ceramic fillers and
loadings that give a capacitive pressure sensor the highest robust capacitance sensitivity over
that range. The app has three pages:

- **Composite Optimizer:** runs the genetic-algorithm search and shows ranked formulations,
  predicted response curves and an experiment plan.
- **Database:** the matrices and fillers the optimizer can choose from, which you can add to or delete from.
- **AI Assistant:** answers questions about the sensors, materials and model from Sensora's own knowledge base.

## Run Locally

Requires Python 3.11 or newer.

```bash
git clone https://github.com/SRD-NTU/Sensora.git
cd Sensora
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

The app requires Supabase for material storage. Configure `SUPABASE_URL` and
`SUPABASE_SECRET_KEY` as described in [supabase/README.md](supabase/README.md).
The AI Assistant page can download a small language model from within the app.
