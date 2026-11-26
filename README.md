# LD-690

**Tamil song poetry, translated as poetry, read by machine.**

LD-690 is a parallel corpus of 690 song lyrics (689 Tamil, one Telugu) from [LyricalDelights.com](https://www.lyricaldelights.com),
each translated stanza by stanza into interpretive English by the site's team since 2011. This repository holds the
public release that accompanies the paper *LD-690, a Parallel Corpus of Tamil Film Song with Poetic English Translations* (Palaniappan, Ravindranath, Sundaramuthy and Ganesan).

**See it live:** every song, the poet-style ring, the meaning map, the poets' family tree and the word trends are
interactive on the [Song Map](https://www.lyricaldelights.com/song-map/). Each dot opens the full translation.

## What is here

| Path | What it is |
|---|---|
| `data/ld690.jsonl` | One record per song: id, URL, title, film or source, song, year, lyricist, singer, composer, mood labels, language (Tamil or Telugu), posting date, and the **English translation** as a list of stanzas. |
| `data/features.csv` | Derived per-song features: Tamil line and token counts, whether the song uses நீ ("you"), mean word length, the held-out lyricist-style guess and confidence, Song Map coordinates (poet style and meaning), discovered theme. |
| `data/bge_m3_vectors.npy`, `data/bge_m3_vector_ids.json` | 1,024-d meaning vectors (bge-m3, Tamil and English averaged), one row per id. |
| `data/collections/ids.json` | Mood labels as used on the site (see the note on moods below). |
| `results/` | Every number in the paper, as JSON (`results/paper/`) and the Song Map's own outputs (`results/songmap/`). |
| `scripts/` | The Song Map analysis code (parsing, lyricist-style model, trends, structure, clustering, embeddings). |
| `scripts/analysis/` | The experiments reported in the paper (corpus statistics, robustness and replication on TLC, structure, meaning checks, what the English translations carry, and the NLLB-200 machine-translation baseline). |
| `data/post_authors.json` | WordPress author id of each post (public API), used to study the translators' voices. |

### What is *not* here, and how to get it

The **Tamil lyrics and their transliterations are not redistributed**: the lyrics are the copyrighted work of their
lyricists and rights holders. They are publicly readable on each song's page (the `url` field), and you can rebuild
the full corpus locally for research from the site's public WordPress API:

```sh
pip install numpy scikit-learn
python3 scripts/fetch_archive.py          # writes data/posts.json, tags.json, categories.json (1 request/second)
python3 scripts/songmap/parse_posts.py    # writes data/songmap/songs.json with Tamil + English per song
```

Please keep to the built-in one request per second: the site runs on a small shared host.
The independent comparison corpus (Chinnappa and Dhandapani, 2021) is at
[praveenraj0904/tamillyricscorpus](https://github.com/praveenraj0904/tamillyricscorpus); clone it into
`paper/ext/tlc` (the path the scripts expect) to re-run `scripts/analysis/robustness.py`.

## Notes on the data

- **Selection.** Songs were hand-picked for the quality of their lyric. LD-690 is an anthology, not a sample of Tamil film song; see the paper's Limitations.
- **Translations.** Interpretive by design: they keep the image rather than gloss it. Written by the team and edited by the founder.
- **Moods.** The ten mood labels were assigned in 2026 with the help of a language model reading the translations, and checked by the team.
- **Lyricist names** are normalised (`Vasana Kavidhai` → Subramanya Bharathi, `Na.Muthukumar` → Na. Muthukumar).

## Licence

- Code (`scripts/`): MIT, see `LICENSE`.
- English translations, metadata, features and vectors (`data/`, `results/`): CC BY-NC-SA 4.0, see `DATA_LICENSE.md`.

## Citation

See `CITATION.cff`. Until the arXiv identifier exists, please cite the repository and the Song Map URL.

## Acknowledgements

We thank Roopa Chozhan for contributing translations to LyricalDelights.com. Supported by API credits from
Anthropic's AI for Science program. Claude (Anthropic) was used for rewriting and
proofreading, and to assist with the analysis code; the authors reviewed all of it.
