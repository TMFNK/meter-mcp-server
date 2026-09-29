# data (private, never shipped)

This folder stays empty in git. Three private paths are needed only
for full local reproduction against human gold labels:

- `data/labels_v1.json` - frozen 60 human labels (30 train, 30 eval).
- `data/raw/` - 8 raw meter CSVs.
- `outputs/label_pack/selection.json` - frozen 30/30 split.

Without them, `reproduce.sh` runs the public path: synth smoke data,
contract tests, STDIO tool list, HTTP health. With them, Step 1 also
runs the sealed eval check (eval-once, results to a copy, never
overwrite).

See the sibling repo `TMFNK/meter-day-classifier` for the data layout.
