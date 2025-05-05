import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd

from .config      import RESULTS_DIR
from .loader      import DataLoader
from .preprocess  import SpacyPreprocessor
from .sentiment   import LexiconSentiment
from .correlation import CountCorrCalculator

class AppController:
    def __init__(self):
        self.loader    = DataLoader()
        self.prep      = SpacyPreprocessor()
        self.sent_an   = LexiconSentiment()
        self.corr_calc = CountCorrCalculator()
        self.df        = self.loader.load()

        self.root = tk.Tk()
        self._build_gui()

    def _build_gui(self):
        self.root.title("Taller Sentimientos y Correlaciones")
        self.root.geometry("650x550")
        self.output = scrolledtext.ScrolledText(self.root, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        frm = tk.Frame(self.root)
        tk.Button(frm, text="1. Anexar Tweets",          command=self.append_tweets).pack(side=tk.LEFT, padx=5)
        tk.Button(frm, text="2. Analizar Sentimientos", command=self.run_sentiment).pack(side=tk.LEFT, padx=5)
        tk.Button(frm, text="3. Correlaciones",         command=self.run_correlations).pack(side=tk.LEFT, padx=5)
        frm.pack(pady=5)

    def append_tweets(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/TSV","*.csv *.txt")])
        if not path: return
        try:
            # lee igual que el loader
            df_new = pd.read_table(
                path, sep="\t", header=None,
                names=["cuenta","partido","timestamp","tweet"],
                encoding="latin-1", dtype=str
            )
            self.df = pd.concat([self.df, df_new], ignore_index=True)
            self.loader.save(self.df)
            messagebox.showinfo("Éxito", f"Se anexaron {len(df_new)} tweets.")
        except Exception as e:
            messagebox.showerror("Error al anexar", str(e))

    def run_sentiment(self):
        self.output.delete(1.0, tk.END)
        try:
            dfp = self.df.copy()
            dfp["clean"]      = dfp["tweet"].apply(self.prep.clean)
            dfp["polarity"]   = dfp["clean"].apply(self.sent_an.polarity)
            dfp["sent_label"] = dfp["polarity"].apply(self.sent_an.label)

            dist = dfp["sent_label"].value_counts(normalize=True).mul(100).round(1)
            self.output.insert(tk.END, "Distribución Sentimientos (%)\n")
            for lbl,pct in dist.items():
                self.output.insert(tk.END, f"  {lbl}: {pct}%\n")

            out_path = f"{RESULTS_DIR}/sentiment_full.csv"
            dfp.to_csv(out_path, index=False, encoding="latin-1")
            self.output.insert(tk.END, f"\n[Guardado en {out_path}]\n")
        except Exception as e:
            messagebox.showerror("Error Sentimiento", str(e))

    def run_correlations(self):
        self.output.delete(1.0, tk.END)
        try:
            dfp = self.df.copy()
            dfp["clean"] = dfp["tweet"].apply(self.prep.clean)
            corr = self.corr_calc.top(dfp["clean"], n=10)
            self.output.insert(tk.END, "Top 10 Correlaciones:\n")
            for (w1,w2),val in corr.items():
                self.output.insert(tk.END, f"  {w1} — {w2}: {val:.2f}\n")

            out_path = f"{RESULTS_DIR}/correlation_full.csv"
            corr.to_csv(out_path, header=["correlation"])
            self.output.insert(tk.END, f"\n[Guardado en {out_path}]\n")
        except Exception as e:
            messagebox.showerror("Error Correlación", str(e))

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    AppController().run()
