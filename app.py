from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
from src.loader      import DataLoader
from src.preprocess  import SpacyPreprocessor
from src.sentiment   import LexiconSentiment
from src.correlation import CountCorrCalculator

app = Flask(__name__, template_folder="templates")
app.secret_key = "cualquier_clave_secreta"  # necesario para flash()

# carga inicial
loader    = DataLoader()
df        = loader.load()

prep      = SpacyPreprocessor()
sent_an   = LexiconSentiment()
corr_calc = CountCorrCalculator()

# ─── RUTA HOME ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ─── RUTA UPLOAD ─────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("tweets_file")
    if not f:
        flash("No seleccionaste ningún fichero", "warning")
        return redirect(url_for("index"))

    try:
        # asumimos TSV con columnas sin cabecera
        df_new = pd.read_table(f.stream,
                               sep="\t",
                               header=None,
                               names=["cuenta","partido","timestamp","tweet"],
                               encoding="latin-1",
                               dtype=str)
        # concatenar y guardar
        global df
        df = pd.concat([df, df_new], ignore_index=True)
        loader.save(df)
        flash(f"Se anexaron {len(df_new)} tweets.", "success")
    except Exception as e:
        flash(f"Error al anexar: {e}", "danger")

    return redirect(url_for("index"))

# ─── RUTA SENTIMENT ──────────────────────────────────────────
@app.route("/sentiment", methods=["POST"])
def sentiment():
    dfp = df.copy()
    dfp["clean"]      = dfp["tweet"].apply(prep.clean)
    dfp["polarity"]   = dfp["clean"].apply(sent_an.polarity)
    dfp["sent_label"] = dfp["polarity"].apply(sent_an.label)

    dist = dfp["sent_label"].value_counts(normalize=True).mul(100).round(1).to_dict()
    examples = {
      lbl: dfp[dfp["sent_label"]==lbl][["tweet","clean","polarity"]]
               .sample(5, random_state=42)
               .values.tolist()
      for lbl in ("neg","pos","neu")
    }
    return render_template("sentiment.html",
                           distribution=dist,
                           examples=examples)

# ─── RUTA CORRELATION ────────────────────────────────────────
@app.route("/correlation", methods=["POST"])
def correlation():
    dfp = df.copy()
    dfp["clean"] = dfp["tweet"].apply(prep.clean)
    corr = corr_calc.top(dfp["clean"], n=10)  # devuelve pd.Series
    return render_template("correlation.html",
                           correlations=corr.items())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
