# Publishing on GitHub

This folder is the source-repository root. The parent manuscript project is a
separate repository. Publish this folder as an independent public repository.

The prepared target is `lixvqing01/pacific-rey-mud`. Create that public repository
before uploading. If another name is chosen, update CITATION.cff and the
download examples, then refresh the source checksums before committing.

1. Choose the repository owner and name. Review the author attribution in
   CITATION.cff and the code/model/data license files. Resolve the article's
   modelling-population statements against docs/PROVENANCE.md.
   Set the actual `repository-code` URL in CITATION.cff before the source commit.
   After intentional edits to existing release files, update and verify their
   checksums:

   ```bash
   python -B tools/refresh_source_checksums.py
   python -B tools/verify_package.py --source-only
   ```

   Keep text files with LF line endings, as specified in .gitattributes, so
   checksums remain identical after Git checkout on Windows and Linux.
2. Create a public GitHub repository and put this folder's regular files in
   it, preserving the directory layout. Upload the files as browsable source,
   including .github, .gitignore and .gitattributes. GitHub Desktop or Git can
   upload the directory tree in one commit. Keep .gitignore: it excludes the six
   large NetCDF assets, local outputs and verification working files. Code,
   tables, split arrays and model weights remain individually browsable.
3. Commit the source and create/push the tag v1.0.1. Retain that tag for the
   submitted version. Do not overwrite the tagged assets with changed data.
4. With GitHub CLI installed and authenticated, first inspect the command:

   ```bash
   python -B tools/publish_github.py --repo lixvqing01/pacific-rey-mud
   ```

   Then publish when ready:

   ```bash
   python -B tools/publish_github.py --repo lixvqing01/pacific-rey-mud --execute
   ```

   Run the publishing tool from the complete local package, where the six
   NetCDF files are present at their paths in release_manifest.json. A source-only
   checkout can instead attach those files manually in the GitHub Release form.
   The tool verifies the local candidate, checks public visibility and creates
   the release against the existing tag, then uploads all six assets. If a
   network transfer is interrupted, finish the missing assets with GitHub CLI;
   verify every file against release_manifest.json.
5. Clone the public repository into a new directory. Install the environment,
   run the CPU example, download inputs/products, run verify_package.py and
   reproduce_evaluation.py. Check access while logged out as well.
6. Put the actual public URL, release tag and commit in the manuscript's Code
   availability and Data availability statements. Record the article DOI in
   a subsequent source version when assigned, refresh its source checksums and
   retain the submitted version's original tag.

The GitHub repository and its versioned Release together host the full workflow.
The journal's repository requirements permit this arrangement; a separate DOI
service is not stated as a requirement in the saved Guide for Authors.
