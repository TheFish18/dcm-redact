# Overview

This tool provides a simple, lightweight way to redact rectangular regions in image files.
It currently supports PNG, TIFF, DICOM (), and other similar image formats.

⚠️ The redaction rectangle must be fully contained within the image boundaries.

## Settings
The following env variables will control various settings: 
- `DCM_REDACT_MAX_DIM`: int, default is 8192. the image being shown will be resized such that the `max(h, w) == DCM_REDACT_MAX_DIM`. Setting this to a smaller value will make rendering faster for larger images.
    - e.g. for linux `export DCM_REDACT_MAX_DIM=2048` for windows powershell `$env:DCM_REDACT_MAX_DIM=2048`, and windows cmd `set DCM_REDACT_MAX_DIM=2048`
- The redacted image will always be saved in full resolution

# DICOM Compatibility

At present, the program supports only DICOM images with the following pixel data attributes:
- `PHOTOMETRIC_INTERPRETATION`: MONOCHROME1 or MONOCHROME2
- `BITS_ALLOCATED`: 16

All output images are saved with:
- `PHOTOMETRIC_INTERPRETATION` = MONOCHROME2
- `BITS_STORED` = 16

# Installation

download binary from [releases page](https://www.github.com/TheFish18/dcm-redact/releases).

