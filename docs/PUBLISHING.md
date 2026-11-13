# Repository and data publication

The public source repository is `lixvqing01/pacific-rey-mud`. It contains the
article's source code, documentation, small datasets, pretrained checkpoints
and runnable CPU example as ordinary files.

The repository is presented as a single source upload without a software
version number. The Data files release hosts the six large NetCDF attachments.
GitHub requires a tag for these attachments; `data` identifies their download
location and is not a software version number.

For intentional maintainer edits, refresh and verify the source checksums:

```bash
python -B tools/refresh_source_checksums.py
python -B tools/verify_package.py --source-only
python -B tools/quick_example.py
```

Keep LF line endings as specified in .gitattributes. Large NetCDF files and
generated outputs are excluded by .gitignore. Existing asset bytes must agree
with the sizes and SHA256 hashes in release_manifest.json.

The optional publishing helper operates on the Data files release:

```bash
python -B tools/publish_github.py --repo lixvqing01/pacific-rey-mud
```

Its default is a dry run. Use `--execute` only when creating that release from
a complete local package; it checks the public repository and local assets
before publishing. The matching `data` tag must already exist. For an existing
release, maintain its attachments through GitHub without creating duplicates.

Verify anonymous repository access and the README example after publication.
Include the public repository URL in the manuscript's Computer Code Availability
statement. Software, model and third-party data terms are stated in LICENSE
and DATA_LICENSE.md.
