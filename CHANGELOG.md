## [1.16.4] - 2023-01-09
### Fixed
- Fix invalid `np.object` type ([#87](https://github.com/xdf-modules/xdf-Python/pull/87) by [Clemens Brunner](https://github.com/cbrnr))
- Fix robust fit for large timestamps ([#73](https://github.com/xdf-modules/xdf-Python/pull/73) by [Chadwick Boulay](https://github.com/cboulay))
- Fix loading stream with no samples when `dejitter_timestamps=False` ([#82](https://github.com/xdf-modules/xdf-python/pull/82) by [Robert Guggenberger](https://github.com/agricolab))

## [1.16.3] - 2020-08-07
### Added
- Add Cython type hints (requires optional local compilation) ([#17](https://github.com/xdf-modules/xdf-python/pull/17) by [Tristan Stenner](https://github.com/tstenner))

### Fixed
- Handle XDF files with corrupt chunk headers (missing stream IDs) more gracefully ([#62](https://github.com/xdf-modules/xdf-python/pull/62) by [Robert Guggenberger](https://github.com/agricolab))

- Treat `nominal_srate` field as float to fix parsing errors ([#65](https://github.com/xdf-modules/xdf-python/pull/62) by [Robert Guggenberger](https://github.com/agricolab) and [#68](https://github.com/xdf-modules/xdf-Python/pull/68) by [Clemens Brunner](https://github.com/cbrnr))

### Changed
- `load_xdf` now requires keyword-only arguments after the first two arguments ([#59](https://github.com/xdf-modules/xdf-python/pull/59) by [Christian Kothe](https://github.com/chkothe))

## [1.16.2] - 2019-10-23
### Added
- Allow loading from already opened file objects, e.g. in-memory files or network streams ([#51](https://github.com/xdf-modules/xdf-python/pull/51) by [Tristan Stenner](https://github.com/tstenner))
- Add CI tests with example data ([#49](https://github.com/xdf-modules/xdf-Python/pull/49) by [Clemens Brunner](https://github.com/cbrnr))

### Fixed
- Compare nominal to effective sampling rates only for regularly sampled streams ([#47](https://github.com/xdf-modules/xdf-Python/pull/47) by [Clemens Brunner](https://github.com/cbrnr))
- More robust error recovery for compressed corrupted files ([#50](https://github.com/xdf-modules/xdf-python/pull/50) by [Tristan Stenner](https://github.com/tstenner))

### Changed
- Speed up loading of numerical data ([#46](https://github.com/xdf-modules/xdf-python/pull/46) by [Tristan Stenner](https://github.com/tstenner))
- Avoid/suppress some NumPy warnings ([#48](https://github.com/xdf-modules/xdf-Python/pull/48) by [Clemens Brunner](https://github.com/cbrnr))

## [1.16.1] - 2019-09-28
### Fixed
- Remove Python 2 compatibility from `setup.py` ([#45](https://github.com/xdf-modules/xdf-Python/pull/45) by [Clemens Brunner](https://github.com/cbrnr))

## [1.16.0] - 2019-09-27
### Added
- Add option to load only specific streams ([#24](https://github.com/xdf-modules/xdf-Python/pull/24) by [Clemens Brunner](https://github.com/cbrnr))
- Add `verbose` argument to `load_xdf` ([#42](https://github.com/xdf-modules/xdf-Python/pull/42) by [Chadwick Boulay](https://github.com/cboulay))

### Fixed
- Fix bug in jitter removal ([#35](https://github.com/xdf-modules/xdf-python/pull/35) by [Alessandro D'Amico](https://github.com/ollie-d))
- Add compatibility with Python 3.5 by converting Pathlike objects to str for file open functions ([#37](https://github.com/xdf-modules/xdf-python/pull/37) by [hankso](https://github.com/hankso))

### Changed
- Refactor jitter removal code to be more readable ([#36](https://github.com/xdf-modules/xdf-python/pull/36) by [Chadwick Boulay](https://github.com/cboulay))

## [1.15.2] - 2019-06-07
### Added
- Store unique stream ID inside the `["info"]["stream_id"]` dict value ([#19](https://github.com/xdf-modules/xdf-Python/pull/19) by [Clemens Brunner](https://github.com/cbrnr))

## [1.15.1] - 2019-04-26
### Added
- Support pathlib objects ([#7](https://github.com/xdf-modules/xdf-Python/pull/7) by [Clemens Brunner](https://github.com/cbrnr))
- Allow example script to be called with an optional XDF file name (e.g. `python -m pyxdf.example` or `python -m pyxdf.example xdf_file.xdf`) ([#10](https://github.com/xdf-modules/xdf-Python/pull/10) by [Tristan Stenner](https://github.com/tstenner))

### Fixed
- Use correct data types ([#2](https://github.com/xdf-modules/xdf-Python/pull/2) by [Tristan Stenner](https://github.com/tstenner))
- Streams are not sorted by name anymore ([#3](https://github.com/xdf-modules/xdf-Python/pull/3) by [Clemens Brunner](https://github.com/cbrnr))
- Support older NumPy versions < 1.14 ([#4](https://github.com/xdf-modules/xdf-Python/pull/4) by [Clemens Brunner](https://github.com/cbrnr))
- Pass correct on_chunk argument ([#5](https://github.com/xdf-modules/xdf-Python/pull/5) by [Clemens Brunner](https://github.com/cbrnr))
- Fix `_scan_forward` ([#6](https://github.com/xdf-modules/xdf-Python/pull/6) by [Clemens Brunner](https://github.com/cbrnr))

### Changed
- Pull `StreamData` class and chunk 3 reader out of `load_xdf` ([#13](https://github.com/xdf-modules/xdf-Python/pull/13) by [Tristan Stenner](https://github.com/tstenner))
