import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

GENRES = [
    "action", "animation", "biography", "comedy", "crime", "drama", "family",
    "fantasy", "horror", "mystery", "romance", "sci-fi", "thriller",
]

ROOT = Path(__file__).resolve().parents[1]

# GloVe: recnik pretreniranih vektora reci (Stanford NLP, 2014). Tekstualni fajl,
# jedan red po reci -- "king 0.1 0.2 ... 0.4", dakle rec pa 300 brojeva koji je
# opisuju. Reci koje se javljaju u slicnim recenicama imaju slicne brojeve.
# 400.000 reci.
# Potreban samo za replikaciju modela iz rada - nasi modeli ga ne koriste.
GLOVE_PATH = ROOT / "data" / "glove" / "glove.6B.300d.txt"



def load_split(split):
    """
        Vraca jedan skup podataka (train/val/test) kao tabelu.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split mora biti train/val/test, dobijeno: {split!r}")
    return pd.read_csv(ROOT / f"{split}_data" / f"{split}_labels.csv")


def image_path(split, movie_id):
    """Vraca putanju do postera jednog filma."""
    return ROOT / f"{split}_data" / "images" / f"{movie_id}.jpg"


def clean_plot(s):
    """Priprema sirov tekst za model"""
    return (s.fillna("")
             .str.replace(r"\{\{[^}]*\}\}", " ", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())



def load_glove(path=None):
    """Ucitava GloVe fajl u memoriju.

    Vraca dva objekta:
        vocab : dict     rec -> broj reda           {"king": 691, ...}
        emb   : ndarray  (400000, 300) float32      emb[691] = vektor reci "king"

    Vektor se dobija u dva koraka -- recnik daje red, matrica brojeve:

        vocab, emb = load_glove()
        emb[vocab["king"]]      ->  array([0.003, -0.346, ..., 0.12])
    """
    path = Path(path or GLOVE_PATH)
    if not path.exists():
        raise FileNotFoundError(f"nema GloVe fajla: {path}\nVidi uputstvo u README.md")

    vocab, vecs = {}, []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            vocab[parts[0]] = i
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
    return vocab, np.vstack(vecs)


def glove_encode(texts, vocab, emb, max_words=3000):
    """Svaki tekst pretvara u 300 brojeva.

    Za svaku rec iz teksta uzme njen GloVe vektor, pa sve te vektore spoji tako sto uzme prosek tih brojeva.
    Od celog teksta ostane jedan red brojeva:

        G = glove_encode(["A zombie attacks the city"], vocab, emb)
        G.shape     ->  (1, 300)

    texts     : lista ili Series tekstova
    vocab, emb: rezultat load_glove()
    max_words : duzi tekstovi se seku (3000 je vrednost iz rada)

    Mana: u proseku sve reci vrede jednako. `the` se u tekstu javi oko 50
    puta, a rec koja odaje zanr obicno jednom - pa `the` vise utice na
    rezultat.
    """
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = re.findall(r"[a-z']+", str(t).lower())[:max_words]
        idx = [vocab[w] for w in toks if w in vocab]
        if idx:
            out[i] = emb[idx].mean(0)
    return out


def evaluate(y_true, y_score, thresholds=None, name=""):
    """Ocenjuje model i vraca recnik sa sedam mera.

    y_true : matrica (broj_filmova, 13), nule i jedinice -- tacni zanrovi
    y_score: matrica istog oblika, verovatnoce iz modela (ne 0/1!)
    pragovi: granica po zanru; ako se ne zada, koristi se 0.5 za sve

    Tacnost (accuracy) ovde ne vredi: 80% matrice su nule, pa model koji uvek
    kaze "nije" ima 80% tacnosti i nula pogodjenih zanrova. Zato AP i F1.
    Rad koristi iste mere, pa su rezultati direktno uporedivi.

    AP (average precision) -- povrsina ispod precision-recall krive. Meri
        koliko dobro model RANGIRA: da li su pravi horori na vrhu liste.
        Ne zavisi od praga, pa je glavna mera za poredjenje modela.
    F1 -- harmonijska sredina preciznosti i odziva, kad je odluka doneta.
        Kaznjava neuravnotezenost: ako je jedno 0.9 a drugo 0.1, F1 je 0.18.
    P (preciznost) -- od svega sto je model rekao "da", koliko je bilo tacno.
    R (odziv) -- od svega sto zaista jeste taj zanr, koliko je model nasao.
        P i R su u suprotnosti i sluze da se vidi KAKO model gresi: visok
        odziv uz nisku preciznost znaci da previse predvidja.

    MICRO / MACRO / SAMPLES -- tri nacina da se 13 zanrova sabere u jedan broj
    micro   : svi parovi (film, zanr) u istom kosu; cesti zanrovi dominiraju
    macro   : racuna se po zanru, pa prosek; svaki zanr vredi isto
    samples : racuna se po filmu, pa prosek
        Macro je najstroza i za nas najvaznija, jer je skup 10x neuravnotezen
        (drama 1806 : animation 175). Model koji ignorise retke zanrove imace
        dobar micro i los macro.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    th = np.full(y_true.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)
    y_pred = (y_score >= th).astype(int)

    return {
        "model": name,
        "AP_micro": average_precision_score(y_true, y_score, average="micro"),
        "AP_macro": average_precision_score(y_true, y_score, average="macro"),
        "AP_samples": average_precision_score(y_true, y_score, average="samples"),
        "F1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "P_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "R_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
    }


def per_genre_ap(y_true, y_score):
    """Daje ocenu modela za svaki zanr posebno, umesto jednog broja za sve.

    Kao djacka knjizica: evaluate() daje prosek, ovo daje ocene po predmetu.

        per_genre_ap(y_true, y_score)
        -> horror 0.87, drama 0.85, ..., biography 0.49    (13 vrednosti)

    Prosek ovih 13 brojeva jednak je AP_macro iz evaluate().

    Sluzi da se vidi gde model stvarno pada. Prosek 0.73 izgleda ujednaceno,
    a zapravo se krije raspon od 0.49 do 0.87.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    return pd.Series(
        {g: average_precision_score(y_true[:, i], y_score[:, i])
         for i, g in enumerate(GENRES)},
        name="AP",
    )


def tune_thresholds(y_true, y_score, grid=None):
    """Bira granicu odlucivanja za svaki zanr posebno.

    Kod vise labela nema pobednicke klase, nego se za svaki zanr odlucuje
    zasebno. Jedinstvena granica od 0.5 je losa kada su klase neizbalansirane,
    jer retki zanrovi nikad ne budu predvidjeni.

    Poziva se iskljucivo nad validacionim skupom. Bilo kakvo podesavanje nad
    test skupom je curenje podataka.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    grid = np.arange(0.05, 0.95, 0.01) if grid is None else grid

    best = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, i], (y_score[:, i] >= t).astype(int),
                           zero_division=0) for t in grid]
        best[i] = grid[int(np.argmax(scores))]
    return best


def results_table(rows):
    """Slaze rezultate vise modela u jednu tabelu, poredjanu od najboljeg."""
    return (pd.DataFrame(rows)
            .set_index("model")
            .round(4)
            .sort_values("AP_macro", ascending=False))


# Objavljeni rezultati iz rada (LeBaron, CS230), za poredjenje sa nasim.
PAPER_RESULTS = pd.DataFrame(
    [["Poster (rad)", 0.520138, 0.446334, 0.657156],
     ["Metadata (rad)", 0.494375, 0.457449, 0.628409],
     ["Video (rad)", 0.589880, 0.573914, 0.696374],
     ["Text (rad)", 0.631661, 0.619510, 0.749681],
     ["Combined (rad)", 0.626956, 0.564063, 0.735774]],
    columns=["model", "AP_micro", "AP_macro", "AP_samples"],
).set_index("model")
