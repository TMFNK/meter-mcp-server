# artifacts

Frozen model outputs. Vendored in Step 1 from the sibling repo
`TMFNK/meter-day-classifier`, byte-identical.

- `model.joblib` - frozen sklearn logreg C=1.0, trained on synth-1080 weak labels.
- `operating_point.json` - frozen threshold t=0.0, 26 medians, feature list, winner params.

No fitting happens in this repo. These files are read-only at runtime.
Regeneration: see the sibling repo `reproduce.sh`. A rerun here must
leave these files byte-identical (checked by md5 in `reproduce.sh`).
