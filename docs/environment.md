# Environment

Copy `configs/toolchains.example.toml` to the ignored `toolchains.toml`, or set:

```bash
export QAIRT_SDK_ROOT=/path/to/qairt
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK
export HEXAGON_TOOLS_ROOT=/path/to/HEXAGON_Tools/19.0.04/Tools
export ANDROID_NDK_ROOT=/path/to/android-ndk
```

`HEXAGON_MLIR_ROOT` and `HEXAGON_MLIR_PYTHON` are optional in prebuilt mode and
required in source mode. The latter must point at the Python interpreter whose
environment contains the Hexagon-MLIR Triton build. The executor derives the
matching `TRITON_SHARED_OPT_PATH`, plugin directories, V69 architecture, and
tool paths; it does not overwrite the checked-in object. `ADB_SERIAL` is used
only for device commands.

The verified build environment uses QAIRT 2.41, Hexagon SDK 6.4, Hexagon Tools 19.0.04, Android NDK r27d, and a V69 target. These versions are evidence, not hard-coded machine paths.
