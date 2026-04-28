"""ZANE XAI Core — ONNX Inference Server with WebSocket API.

Provides :class:`ONNXInferenceServer`, a WebSocket-based inference gateway
that:

1. Loads (or exports) a surrogate model as an ONNX graph.
2. Receives JSON payloads ``{"smiles": str}`` over a WebSocket connection.
3. Converts the SMILES to a 1024-bit Morgan fingerprint (via RDKit) as the
   input tensor.
4. Runs **Monte Carlo Dropout** (10 stochastic forward passes) to estimate
   epistemic uncertainty.
5. Returns ``{"mean_score": float, "variance": float, "confidence_warning": bool}``
   in under 150 ms on CPU.

Usage example (programmatic)::

    import asyncio
    server = ONNXInferenceServer()
    server.load()
    result = server.predict("CCO")          # sync convenience method
    # or start the WebSocket server:
    asyncio.run(server.serve("0.0.0.0", 8765))

The model exported here is a small two-hidden-layer MLP with Dropout (p=0.15).
Replace :meth:`ONNXInferenceServer._build_torch_model` with your trained
surrogate and call :meth:`ONNXInferenceServer.export` once.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore[import-untyped]
    import torch.nn as nn

    _TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH = False

try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _ORT = True
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]
    _ORT = False
    logger.error("onnxruntime not installed — ONNXInferenceServer will not function.")

try:
    import websockets  # type: ignore[import-untyped]

    _WEBSOCKETS = True
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore[assignment]
    _WEBSOCKETS = False
    logger.warning("websockets not installed — serve() method will not function.")

try:
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import AllChem  # type: ignore[import-untyped]

    _RDKIT = True
except ImportError:  # pragma: no cover
    _RDKIT = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_INPUT_DIM: int = 1024       # Morgan fingerprint length
_HIDDEN_DIM: int = 128
_OUTPUT_DIM: int = 1
_DROPOUT_P: float = 0.15
_MC_SAMPLES: int = 10
_CONFIDENCE_THRESHOLD: float = 0.1  # variance above this triggers warning
_DEFAULT_MODEL_PATH: str = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "zane_xai_surrogate.onnx"
)


# ---------------------------------------------------------------------------
# PyTorch surrogate (exported to ONNX once; kept in training-mode for MC Dropout)
# ---------------------------------------------------------------------------
def _build_torch_model() -> Any:
    """Construct the lightweight surrogate MLP with Dropout layers.

    Replace this with your trained model when deploying for real.
    """
    if not _TORCH:
        raise RuntimeError("PyTorch is required to export the ONNX surrogate model.")

    class _Surrogate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(_INPUT_DIM, _HIDDEN_DIM)
            self.drop1 = nn.Dropout(p=_DROPOUT_P)
            self.fc2 = nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM // 2)
            self.drop2 = nn.Dropout(p=_DROPOUT_P)
            self.out = nn.Linear(_HIDDEN_DIM // 2, _OUTPUT_DIM)

        def forward(self, x: Any) -> Any:  # type: ignore[override]
            x = torch.relu(self.fc1(x))
            x = self.drop1(x)
            x = torch.relu(self.fc2(x))
            x = self.drop2(x)
            return self.out(x)

    return _Surrogate()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _smiles_to_fp(smiles: str, n_bits: int = _INPUT_DIM) -> np.ndarray:
    """Convert a SMILES string to a Morgan fingerprint (float32 array).

    Args:
        smiles: Input SMILES.
        n_bits: Fingerprint length.

    Returns:
        Float32 numpy array of shape ``(1, n_bits)``.
    """
    if _RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            return np.array(fp, dtype=np.float32).reshape(1, -1)

    # Deterministic fallback (no RDKit / parse failure)
    from hashlib import sha256

    digest = sha256(smiles.encode()).digest()
    arr = np.frombuffer(digest * (n_bits // 32 + 1), dtype=np.uint8)[:n_bits].astype(np.float32) / 255.0
    return arr.reshape(1, -1)


# ---------------------------------------------------------------------------
# Main server class
# ---------------------------------------------------------------------------
class ONNXInferenceServer:
    """WebSocket ONNX inference server with Monte Carlo Dropout uncertainty.

    Args:
        model_path: Path to the ONNX model file.  If it does not exist,
            :meth:`export` is called automatically on first :meth:`load`.
        mc_samples: Number of stochastic forward passes for MC Dropout.
        confidence_threshold: Variance threshold above which
            ``confidence_warning`` is set to ``True``.

    Example::

        server = ONNXInferenceServer()
        server.load()
        result = server.predict("c1ccccc1")
        # {"mean_score": -0.42, "variance": 0.003, "confidence_warning": False}
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        mc_samples: int = _MC_SAMPLES,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_path = model_path
        self.mc_samples = mc_samples
        self.confidence_threshold = confidence_threshold
        self._session: Any = None  # onnxruntime.InferenceSession

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def export(self) -> None:
        """Export the PyTorch surrogate to ONNX (training mode for MC Dropout).

        Only required once.  The file is saved to :attr:`model_path`.
        """
        if not _TORCH:
            raise RuntimeError("PyTorch is required to export the ONNX model.")

        model = _build_torch_model()
        model.train()  # Keep dropout active in ONNX graph

        dummy = torch.zeros(1, _INPUT_DIM)
        os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            self.model_path,
            input_names=["fingerprint"],
            output_names=["score"],
            dynamic_axes={"fingerprint": {0: "batch"}, "score": {0: "batch"}},
            opset_version=17,
            training=torch.onnx.TrainingMode.TRAINING,  # type: ignore[attr-defined]
            do_constant_folding=False,
        )
        logger.info("ONNX surrogate exported to %r", self.model_path)

    def load(self) -> None:
        """Initialise the :class:`onnxruntime.InferenceSession`.

        Exports the ONNX model first if :attr:`model_path` does not exist.
        """
        if not _ORT:
            raise RuntimeError("onnxruntime is required but not installed.")

        if not os.path.exists(self.model_path):
            logger.info("ONNX model not found at %r — exporting now.", self.model_path)
            self.export()

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNXInferenceServer ready (model=%r, mc_samples=%d)", self.model_path, self.mc_samples)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._session is None:
            self.load()

    def predict(self, smiles: str) -> dict[str, Any]:
        """Run MC Dropout inference for a single SMILES.

        Args:
            smiles: Input SMILES string.

        Returns:
            ``{"mean_score": float, "variance": float, "confidence_warning": bool}``
        """
        self._ensure_loaded()

        fp = _smiles_to_fp(smiles)  # shape (1, INPUT_DIM)
        scores: list[float] = []

        input_name = self._session.get_inputs()[0].name  # type: ignore[union-attr]
        for _ in range(self.mc_samples):
            outputs = self._session.run(None, {input_name: fp})  # type: ignore[union-attr]
            scores.append(float(outputs[0][0, 0]))

        mean_score = float(np.mean(scores))
        variance = float(np.var(scores))
        confidence_warning = variance > self.confidence_threshold

        return {
            "mean_score": mean_score,
            "variance": variance,
            "confidence_warning": confidence_warning,
        }

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------
    async def websocket_handler(self, websocket: Any) -> None:
        """Async WebSocket handler.

        Accepts JSON messages of the form ``{"smiles": "<SMILES>"}`` and
        replies with ``{"mean_score": float, "variance": float,
        "confidence_warning": bool}``.

        Designed to complete inference in under 150 ms on CPU.

        Args:
            websocket: A ``websockets`` connection object.
        """
        async for raw_message in websocket:
            t_start = time.perf_counter()
            try:
                payload = json.loads(raw_message)
                smiles = payload.get("smiles", "")
                if not smiles:
                    await websocket.send(json.dumps({"error": "Missing 'smiles' field"}))
                    continue

                result = self.predict(smiles)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                result["latency_ms"] = round(elapsed_ms, 2)

                if elapsed_ms > 150:
                    logger.warning("Inference exceeded 150 ms budget: %.1f ms", elapsed_ms)

                await websocket.send(json.dumps(result))

            except json.JSONDecodeError as exc:
                await websocket.send(json.dumps({"error": f"Invalid JSON: {exc}"}))
            except Exception as exc:
                logger.error("Inference error: %s", exc)
                await websocket.send(json.dumps({"error": str(exc)}))

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------
    async def serve(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """Start the WebSocket server.

        Args:
            host: Bind address.
            port: TCP port.
        """
        if not _WEBSOCKETS:
            raise RuntimeError("websockets package is required to call serve().")

        self._ensure_loaded()
        logger.info("Starting ONNXInferenceServer on ws://%s:%d", host, port)
        async with websockets.serve(self.websocket_handler, host, port):  # type: ignore[attr-defined]
            await asyncio.Future()  # run forever
