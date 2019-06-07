## [1.15.2] - 2019-06-07
### Added
- Store unique stream ID inside the `["info"]["stream_id"]` dict value ([#19](https://github.com/xdf-modules/xdf-Python/pull/19) by [Clemens Brunner](https://github.com/cbrnr)).

## [1.15.1] - 2019-04-26
### Added
- Support pathlib objects ([#7](https://github.com/xdf-modules/xdf-Python/pull/7) by [Clemens Brunner](https://github.com/cbrnr)).
- Allow example script to be called with an optional XDF file name (e.g. `python -m pyxdf.example` or `python -m pyxdf.example xdf_file.xdf`) ([#10](https://github.com/xdf-modules/xdf-Python/pull/10) by [Tristan Stenner](https://github.com/tstenner)).

### Fixed
- Use correct data types ([#2](https://github.com/xdf-modules/xdf-Python/pull/2) by [Tristan Stenner](https://github.com/tstenner)).
- Streams are not sorted by name anymore ([#3](https://github.com/xdf-modules/xdf-Python/pull/3) by [Clemens Brunner](https://github.com/cbrnr)).
- Support older NumPy versions < 1.14 ([#4](https://github.com/xdf-modules/xdf-Python/pull/4) by [Clemens Brunner](https://github.com/cbrnr)).
- Pass correct on_chunk argument ([#5](https://github.com/xdf-modules/xdf-Python/pull/5) by [Clemens Brunner](https://github.com/cbrnr)).
- Fix _scan_forward ([#6](https://github.com/xdf-modules/xdf-Python/pull/6) by [Clemens Brunner](https://github.com/cbrnr)).

### Changed
- Pull `StreamData` class and chunk 3 reader out of `load_xdf` ([#13](https://github.com/xdf-modules/xdf-Python/pull/13) by [Tristan Stenner](https://github.com/tstenner)).
