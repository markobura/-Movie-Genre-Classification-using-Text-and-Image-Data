#!/usr/bin/env python3
"""
Deli MovieScope dataset na train_data/, val_data/ i test_data/.

Originalni podaci nisu fizicki podeljeni -- posteri i metapodaci stoje svi
zajedno, a podela postoji samo kao spisak movie_id-jeva unutar .p fajlova.
Ova skripta tu podelu materijalizuje u foldere.

Ulaz:   data/movie_scope/
Izlaz:  {split}_data/raw_data_{split}.p
        {split}_data/images/{movie_id}.jpg
        {split}_data/movie_metadata_{split}.csv
        {split}_data/{split}_labels.csv      plot + labele, citljivo

Pokretanje:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --force    # prepisi postojece foldere
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd

GENRES = [
    "action", "animation", "biography", "comedy", "crime", "drama", "family",
    "fantasy", "horror", "mystery", "romance", "sci-fi", "thriller",
]

SPLITS = ["train", "val", "test"]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "movie_scope"
POSTERS = SRC / "poster_images" / "MatchedPosters"


def load_pickle(split):
    with open(SRC / "trailers" / f"raw_data_{split}.p", "rb") as f:
        # latin1 jer su .p fajlovi napravljeni u Python-u 2
        return pickle.load(f, encoding="latin1")


def fix_encoding(text):
    """Popravlja mojibake: 'mineral â\\x80\\x93 unobtanium' -> 'mineral – unobtanium'.

    Tekst je UTF-8, ali ga latin1 ucitavanje tumaci bajt-po-bajt. Pogadja ~34% plotova.
    """
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def labels_to_columns(bits):
    """'1000100000001' -> {'action': 1, 'animation': 0, ...}"""
    if len(bits) != len(GENRES):
        raise ValueError(f"ocekivano {len(GENRES)} karaktera, dobijeno: {bits!r}")
    return {g: int(b) for g, b in zip(GENRES, bits)}


def build_split(split, records, metadata, force):
    outdir = ROOT / f"{split}_data"
    imgdir = outdir / "images"

    if outdir.exists() and any(outdir.iterdir()):
        if not force:
            sys.exit(f"GRESKA: {outdir.name}/ vec postoji. Pokreni sa --force.")
        shutil.rmtree(outdir)
    imgdir.mkdir(parents=True, exist_ok=True)

    ids = [r["movie_id"] for r in records]

    shutil.copy2(SRC / "trailers" / f"raw_data_{split}.p",
                 outdir / f"raw_data_{split}.p")

    missing = []
    for mid in ids:
        src_img = POSTERS / f"{mid}.jpg"
        if src_img.exists():
            shutil.copy2(src_img, imgdir / f"{mid}.jpg")
        else:
            missing.append(mid)

    md_split = (metadata[metadata["movie_id"].isin(ids)]
                .sort_values("movie_id").reset_index(drop=True))
    md_split.to_csv(outdir / f"movie_metadata_{split}.csv", index=False)

    rows = []
    for r in records:
        row = {"movie_id": r["movie_id"], "plot": fix_encoding(r["plot"])}
        row.update(labels_to_columns(r["newGenreLabels"]))
        row["genres"] = "|".join(g for g in GENRES if row[g] == 1)
        rows.append(row)
    labels_df = pd.DataFrame(rows).sort_values("movie_id").reset_index(drop=True)
    labels_df.to_csv(outdir / f"{split}_labels.csv", index=False)

    print(f"  {split:5s} -> {outdir.name}/  {len(ids)} filmova | "
          f"{len(list(imgdir.glob('*.jpg')))} slika | {len(md_split)} redova metapodataka"
          + (f" | FALI SLIKA: {len(missing)}" if missing else ""))

    return {"ids": set(ids), "missing_images": missing, "labels": labels_df}


def verify(results):
    print("\n=== PROVERE ===")
    ok = True

    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            sa, sb = SPLITS[a], SPLITS[b]
            overlap = results[sa]["ids"] & results[sb]["ids"]
            ok &= not overlap
            print(f"  [{'OK' if not overlap else f'PAO ({len(overlap)})':18s}] "
                  f"{sa} i {sb} nemaju zajednickih filmova")

    for s in SPLITS:
        miss = results[s]["missing_images"]
        ok &= not miss
        print(f"  [{'OK' if not miss else f'PAO ({len(miss)})':18s}] {s}: svi filmovi imaju poster")

    for s in SPLITS:
        n_img = len(list((ROOT / f"{s}_data" / "images").glob("*.jpg")))
        n_exp = len(results[s]["ids"])
        ok &= n_img == n_exp
        print(f"  [{'OK' if n_img == n_exp else 'PAO':18s}] {s}: {n_img} slika za {n_exp} filmova")

    for s in SPLITS:
        md = pd.read_csv(ROOT / f"{s}_data" / f"movie_metadata_{s}.csv")
        same = set(md["movie_id"]) == results[s]["ids"]
        ok &= same
        print(f"  [{'OK' if same else 'PAO':18s}] {s}: movie_id u metapodacima odgovara splitu")

    # Hvata pomeranje redova pri filtriranju -- movie_id je indeks reda u originalu.
    orig = pd.read_csv(SRC / "movie_metadata.csv")
    for s in SPLITS:
        md = pd.read_csv(ROOT / f"{s}_data" / f"movie_metadata_{s}.csv")
        sample = md.sample(min(50, len(md)), random_state=0)
        bad = [r.movie_id for r in sample.itertuples()
               if str(orig.iloc[int(r.movie_id)]["movie_title"]) != str(r.movie_title)]
        ok &= not bad
        print(f"  [{'OK' if not bad else f'PAO ({len(bad)})':18s}] {s}: naslovi se poklapaju sa originalom")

    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="prepisi postojece foldere")
    args = ap.parse_args()

    if not POSTERS.exists():
        sys.exit(f"GRESKA: nema foldera sa posterima: {POSTERS}")

    # movie_metadata.csv nema kolonu sa ID-em -- identitet filma je BROJ REDA.
    # Bez ovoga bi filtriranje redova trajno pokidalo vezu sa posterima.
    metadata = pd.read_csv(SRC / "movie_metadata.csv")
    metadata.insert(0, "movie_id", metadata.index)

    print(f"Ucitano {len(metadata)} redova iz movie_metadata.csv")
    print(f"Postera na disku: {len(list(POSTERS.glob('*.jpg')))}\n")
    print("Pravim foldere:")

    results = {}
    for split in SPLITS:
        results[split] = build_split(split, load_pickle(split), metadata, args.force)

    ok = verify(results)

    print("\n=== ZANROVI PO SPLITOVIMA (broj filmova) ===")
    dist = pd.DataFrame({s: results[s]["labels"][GENRES].sum() for s in SPLITS})
    print(dist.to_string())

    # 109 filmova iz CSV-a ne zavrsi ni u jednom splitu: ~91 nema nijedan od 13
    # zanrova (labela bi bila sve nule), za ostalih ~18 razlog nije dokumentovan.
    total = sum(len(results[s]["ids"]) for s in SPLITS)
    izbaceni = metadata[~metadata["movie_id"].isin(
        set().union(*(results[s]["ids"] for s in SPLITS)))]
    bez_13 = sum(1 for g in izbaceni["genres"]
                 if not {x.lower() for x in str(g).split("|")} & set(GENRES))
    print(f"\nRasporedjeno: {total} filmova. Van splitova: {len(izbaceni)} "
          f"({bez_13} bez ijednog od 13 zanrova, {len(izbaceni) - bez_13} bez poznatog razloga).")

    if not ok:
        sys.exit("\nNEKE PROVERE NISU PROSLE -- ne koristi ove podatke.")
    print("\nSve provere prosle.")


if __name__ == "__main__":
    main()
