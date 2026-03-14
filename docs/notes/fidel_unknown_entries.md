# FIDEL Unknown Archive Entries

## Summary

During `extract-fidel`, some PNG files are counted as `unknown`. These are not extraction failures. They are archive entries that do not resolve to any trusted catalog row in the upstream FIDEL metadata.

For the sampled unknowns checked here, the issue is upstream catalog coverage, not our extracted filename prefixing or lookup logic.

## What `unknown` Means

`extract-fidel` matches archive PNGs against the upstream catalogs by:

- source repository
- source split
- original filename

If a PNG exists in the archive but no catalog row matches those fields, it is counted as `unknown` and skipped from the extracted training snapshot.

## Counts Observed

The sampled investigation found these `unknown` counts:

- `fidel_dataset/train`: `591`
- `fidel_dataset/test`: `16581`
- `fidel_synthetic/synthetic`: `75`

## Sampled Unknown Filenames

Examples checked directly from the raw archives:

- `hand_642_line_8.png`
- `hand_1400_line_11.png`
- `hand_453_line_11.png`

## Verification Performed

The sampled unknown filenames were checked against:

- `input/ocr_training/fidel/raw/fidel_dataset/train_labels.csv`
- `input/ocr_training/fidel/raw/fidel_dataset/test_labels.csv`
- `input/ocr_training/fidel/raw/fidel_synthetic/synthetic_labels.csv`

For the sampled cases, the filenames were absent from all three catalogs.

## Conclusion

For the examples verified here, the `unknown` entries are genuinely uncataloged upstream files. They are not caused by:

- our normalized extracted filename prefixing
- our downstream renamed image paths
- a mismatch in the extracted snapshot naming convention

The extraction pipeline is correctly skipping them because it cannot attach them to a trusted metadata row.

## Implication

These entries should be treated as unlabeled or unindexed source files, not automatically as mislabeled OCR rows. They are a separate upstream data-quality issue from:

- blank-image rows with non-empty transcriptions
- text-image label mismatches
- artifact-heavy crops with incorrect transcriptions
