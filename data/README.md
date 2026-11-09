# data/

This folder is empty on purpose. The dataset is **not** redistributed here.

## Download

1. Dataset: *Replication Data for: A Trainable Spaced Repetition Model for Language Learning*
   (Settles & Meeder), Harvard Dataverse, version 1.0 (2017-12-14), DOI `10.7910/DVN/N8XJME`.
2. File: `settles.acl16.learning_traces.13m.csv.gz` (Dataverse file id 3091087, 379,004,009 bytes).
3. Expected MD5: `0a1cae5eb7ad4b0bd9c0de91d74fcced`

```bash
curl -L -o data/settles.acl16.learning_traces.13m.csv.gz \
  https://dataverse.harvard.edu/api/access/datafile/3091087
md5 data/settles.acl16.learning_traces.13m.csv.gz      # macOS (Linux: md5sum)
gzip -t data/settles.acl16.learning_traces.13m.csv.gz
```

The loader (`src/data/duolingo.js`) expects exactly that file name in this folder.

## Generated files (all gitignored)

| File | Produced by |
|---|---|
| `inventory.json` | `node src/data/inventory.js` |
| `sample.csv` | `node src/data/sample.js <total_rows> <target_rows>` (the reports use a 1M-row sample) |
| `b0-results.json` … `b3c2-results.json`, `*-diag.json` | `node src/model/b*.js` |

Row-level files derived from the dataset (samples, predictions) must stay local.

## License

The data is licensed **CC BY-NC 4.0** (attribution, non-commercial). Cite:

> Burr Settles and Brendan Meeder. 2016. *A Trainable Spaced Repetition Model for Language Learning.*
> In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL), pages 1848–1858.

The reference implementation is `github.com/duolingo/halflife-regression` (MIT), commit `0041df0`.
