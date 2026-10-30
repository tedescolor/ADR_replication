# Random Wikipedia articles: an i.i.d. dataset

`generate_dataset.py` builds a fresh snapshot of `n` (default 1000) articles drawn at random from English Wikipedia (en.wikipedia.org; other editions via `--lang`), with six covariates and a bivariate response.

```
pip install requests
python3 generate_dataset.py -n 1000 -o dataset.csv --contact you@example.org
```

The script writes two files:

Wikimedia allows only 10 requests per minute without contact information in the User-Agent and 200 per minute with it, so pass `--contact`. An interrupted run resumes from `dataset_sample.json` and `dataset_targets.jsonl`.

- `dataset.csv` is the file to distribute. It has no identifiers and neutral column names (`x1`–`x6`, `y1`, `y2`).
- `dataset_key.csv` is private. It has the same rows in the same order, with page ids, titles, descriptive column names and the untransformed responses.

## Sampling design

Each row comes from a separate call to the MediaWiki API module `list=random` (`rnnamespace=0`, `rnfilterredir=nonredirects`, `rnlimit=1`). Main-namespace articles only, no redirects.

- **Independence.** Every draw uses a fresh server-side random number, so the draws are mutually independent.
- **Identical distribution.** Every draw targets the same population (all non-redirect articles at snapshot time) through the same mechanism.
- **With replacement.** The same article can be drawn more than once. With about 7M articles and n = 1000, the expected number of repeated pairs is about 0.07.
- **Note on the sampling distribution.** `list=random` returns the first article whose stored random key is ≥ a uniform draw. Inclusion probabilities are therefore proportional to the gap between consecutive keys, which makes them close to uniform but not exactly uniform. This affects only *which* distribution the rows are drawn from. The i.i.d. property is unaffected.

All time-dependent quantities use one fixed reference date: UTC midnight of the day the script is run.

## Variables

| Masked | Column in key file | Role | Definition |
|---|---|---|---|
| – | `page_id`, `title` | identifiers | MediaWiki page id and title |
| `x1` | `n_categories` | covariate | number of visible (non-hidden) categories |
| `x2` | `n_internal_links` | covariate | number of links to other main-namespace articles |
| `x3` | `n_languages` | covariate | number of interlanguage links (other Wikipedias with the article) |
| `x4` | `has_image` | covariate | 1 if the article has a lead image (PageImages), else 0 |
| `x5` | `age_years` | covariate | years from the article's first revision to the reference date |
| `x6` | `length_kb` | covariate | current wikitext size in KiB |
| – | `views_30d` | response 1 (raw) | user (non-bot) pageviews, all access methods, over the 30 days before the reference date |
| – | `edits_12m` | response 2 (raw) | number of revisions in the 365 days before the reference date |
| `y1` | `log_views_30d` | response 1 | `log(1 + views_30d)` |
| `y2` | `log_edits_12m` | response 2 | `log(1 + edits_12m)` |

Sources: MediaWiki Action API (`en.wikipedia.org/w/api.php`) and the Wikimedia Pageviews REST API. Both responses cover a period ending on the reference date, so they postdate any pretrained model's training data.
