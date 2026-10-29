# Device deployment

Device artifacts are generated under ignored
`deploy/generated/<plugin>/<target>/bundle/` and pushed below
`/data/local/tmp/hyexec/<plugin>/<target>/`. The bundle contains the model,
ARM and DSP OpPackage libraries, inputs/references, `qnn-net-run`, and the
target-specific QAIRT runtime libraries. It also contains a SHA-256 manifest.

```bash
./hyexec bundle vector_add_mlir --target v69
./hyexec deploy vector_add_mlir --target v69
./hyexec run vector_add_mlir --target v69
./hyexec compare vector_add_mlir --target v69
```

Each run writes a new UTC-stamped output directory. `compare` pulls that
directory without deleting it and checks native outputs against the generated
reference using the manifest tolerances. Device serials and developer-specific
absolute paths are not stored in tracked manifests.
