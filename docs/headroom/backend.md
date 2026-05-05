# Forcing the Kompress backend

`MagosSettings.kompress_backend` (env: `MAGOS_KOMPRESS_BACKEND`)
controls which Kompress backend Headroom uses:

| Value         | Behaviour                                                                 |
|---------------|---------------------------------------------------------------------------|
| `auto` (default) | Headroom prefers ONNX Runtime when installed, falls back to PyTorch. INT8 ONNX runs CPU-only out of the box (Headroom hardcodes `providers=["CPUExecutionProvider"]`). |
| `pytorch`     | Forces the PyTorch backend. `_load_kompress_pytorch` auto-selects CUDA / MPS / CPU via `device='auto'`. This is the path to choose for GPU acceleration. |

Implementation: when set to `pytorch`, the FastAPI lifespan hook
replaces `headroom.transforms.kompress_compressor._is_onnx_available`
with a False-returning stub. Headroom's `_load_kompress` resolves that
name from the module namespace at call time, so the override flips
backend selection without patching Headroom itself. See
`_force_kompress_pytorch` in `ingress/http/lifespan.py`.

Caveats:

- The override is process-wide. Per-rule backend selection isn't
  available because Kompress weights are cached at module level keyed
  by `model_id`, not by backend.
- `pytorch` requires `torch` (and `safetensors` + `transformers`) to
  be installed. If they're missing, the first compress request raises
  `ImportError` from Headroom; magos's lazy import catch logs
  `compress.import_failed` and the rule no-ops.
- For GPU, you also need a CUDA-enabled `torch` build. The default
  PyPI `torch` wheels include CPU + CUDA on Linux/Windows; macOS
  builds ship MPS for Apple Silicon. Check `torch.cuda.is_available()`
  to verify GPU availability.
- The override fires unconditionally at lifespan startup when
  `kompress_backend=pytorch`, regardless of whether any rule actually
  uses `compress`. The cost is one attribute assignment: no I/O, no
  model load.

Why we don't expose ONNX CUDA via this knob: even with `onnxruntime-gpu`
installed, Headroom's `_load_kompress_onnx` hardcodes
`providers=["CPUExecutionProvider"]` (`kompress_compressor.py:179-183`),
so flipping `_is_onnx_available` doesn't help; the ONNX session would
still be CPU-bound. A working ONNX-CUDA path needs an upstream Headroom
patch to thread an EP list through, plus careful handling of INT8
operator coverage on CUDA EP (many INT8 ops fall back to CPU). See
prior research notes; not pursued.
