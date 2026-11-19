# Dataset and processing notes

`raw_data/weatherAUS.csv` is the course-supplied input and must remain unchanged. The independent cleaning script is `src/clean_data.py`; it writes `cleaned_data/weatherAUS_clean.csv` and a provenance/cleaning record at `cleaned_data/cleaning_log.json`.

The supplied file has 145,460 rows and 23 columns. It covers 2007-11-01 through 2017-06-25. Cleaning removes 3,267 rows without a valid `RainTomorrow` label (there are no invalid dates and no exact duplicate labeled records in this snapshot), leaving 142,193 rows. Original counts: 110,316 `No`, 31,877 `Yes`.

Text whitespace is trimmed and blank-like tokens are normalized. `Date` is parsed and serialized as ISO dates. The target is standardized to `No`/`Yes`. Features with missing values remain in the cleaned file; model-pipeline imputers are fitted only on training folds. Statistical outliers are flagged for review but are not deleted or clipped.

The project treats the CSV as a fixed course-supplied snapshot. The Australian Bureau of Meteorology describes daily weather observations at [bom.gov.au/climate/dwo](https://www.bom.gov.au/climate/dwo/), but the archive supplied here does not independently establish the exact extraction/version or redistribution license. Verify citation and sharing permission with the course instructor before publishing the CSV.
