#!/usr/bin/env bash
set -e

# Usage: ./build_fasterliveportrait_trt.sh <input_dir> <onnx_dir> <trt_output_dir>
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <input_dir> <onnx_models_dir> <trt_output_dir>"
  exit 1
fi

INPUT_DIR="$1"
ONNX_DIR="$2"
TRT_OUTPUT_DIR="$3"

export TensorRT_ROOT=/opt/TensorRT-10.12.0.36/targets/x86_64-linux-gnu
export LD_LIBRARY_PATH=$TensorRT_ROOT/lib:$LD_LIBRARY_PATH

PLUGIN_DIR="$INPUT_DIR/grid-sample3d-trt-plugin"
FLP_DIR="$INPUT_DIR/FasterLivePortrait"

echo "🔵 Cloning required repositories..."
if [ ! -d "$PLUGIN_DIR/.git" ]; then
    git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin.git "$PLUGIN_DIR"
else
    echo "✅ $PLUGIN_DIR already exists, skipping clone."
fi

FLP_JUST_CLONED=false
if [ ! -d "$FLP_DIR/.git" ]; then
    git clone --branch vbrealtime_upgrade https://github.com/varshith15/FasterLivePortrait.git "$FLP_DIR"
    FLP_JUST_CLONED=true
else
    echo "✅ $FLP_DIR already exists, skipping clone."
fi

# Build grid-sample3d plugin
echo "🔵 Building grid-sample3d plugin..."
/workspace/ComfyUI/custom_nodes/ComfyUI-FasterLivePortrait/scripts/build_grid_sample3d_plugin.sh "$PLUGIN_DIR"

# Ensure python symlink (for Docker environments missing it)
ln -sf "$(which python3)" /usr/local/bin/python

# Prepare FasterLivePortrait repo
echo "🔵 Preparing FasterLivePortrait..."
cd "$FLP_DIR"
if [ "$FLP_JUST_CLONED" = true ]; then
    git checkout vbrealtime_upgrade
else
    echo "✅ FasterLivePortrait already exists, skipping git checkout."
fi

# Patch libgrid_sample_3d_plugin.so path
sed -i "/if platform.system().lower() == 'linux':/{n;s|.*|        ctypes.CDLL(\"$PLUGIN_DIR/build/libgrid_sample_3d_plugin.so\", mode=ctypes.RTLD_GLOBAL)|}" "$FLP_DIR/scripts/onnx2trt.py"

# Convert ONNX models to TensorRT
echo "🔵 Running ONNX -> TensorRT conversion..."
PYTHON="$FLP_DIR/scripts/onnx2trt.py"

for MODEL in \
    warping_spade-fix.onnx \
    landmark.onnx \
    motion_extractor.onnx \
    retinaface_det_static.onnx \
    face_2dpose_106_static.onnx \
    appearance_feature_extractor.onnx \
    stitching.onnx \
    stitching_eye.onnx \
    stitching_lip.onnx
do
    if [[ "$MODEL" == "motion_extractor.onnx" ]]; then
        python "$PYTHON" -o "$ONNX_DIR/$MODEL" -p fp32
    else
        python "$PYTHON" -o "$ONNX_DIR/$MODEL"
    fi
done

# Also move the compiled plugin .so
mv "$PLUGIN_DIR/build/libgrid_sample_3d_plugin.so" "$TRT_OUTPUT_DIR"

echo "🎉 All done!"
