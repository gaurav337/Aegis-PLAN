# AEGIS-X IMPLEMENTATION PLAN: PART 3

## PHASE 3: GPU Forensic Tools (Days 11–14)

*(Note: All GPU tools MUST utilize the `VRAMLifecycleManager` defined in Day 5 to prevent OOM errors on 4GB hardware.)*

### Day 11: Universal Forgery Tool (CLIP + Adapter)

#### Prompt for Day 11:

**Section A: Context Reminder**
Aegis-X is an offline, agentic multi-modal forensic engine. 
This is Phase 3, Day 11. We are building the first GPU tool: The CLIP-based Universal Forgery Detector. Because fully-synthetic faces (e.g., from Sora or Midjourney) lack the "blend boundaries" of face-swaps, we rely on CLIP's broad representation (learned from 400M real images) paired with a lightweight forensic adapter.

**Section B: Today's Objectives**
- Create `core/tools/clip_adapter_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**

This is the most architecturally complex tool in Aegis-X. It consists of a frozen CLIP ViT-B/32 backbone (OpenAI, ~150M params, ~600MB) paired with a lightweight trainable forensic adapter (~993K params, ~3.8MB). **The adapter is NOT a simple MLP** — it is a 4-stage pipeline implemented across 5 sub-files.

**File Structure:** Create a `core/tools/clip_adapter/` sub-package:
- `__init__.py`
- `landmark_crops.py` — Stage 0: 6 anatomical crop extraction
- `patch_extractor.py` — Stage 1: CLIP hook extraction (layers 3,6,9,11)
- `bottleneck.py` — Stage 2: spatial pooling + layer fusion
- `attention_head.py` — Stage 3: cross-patch attention + LSE pooling
- `tta.py` — Stage 4: test-time augmentation

**Entry point:** `core/tools/clip_adapter_tool.py`:
- `ClipAdapterTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_clip_adapter"`
  - `def setup(self)`: No-op. Model loading is dynamic to save VRAM.
  - `def _load_model(self) -> tuple`:
    - Import `clip` and load `clip.load("ViT-B/32", device="cpu")`.
    - Freeze CLIP backbone (`param.requires_grad = False` for all params).
    - Initialize the forensic adapter (Bottleneck + AttentionHead modules).
    - Load adapter weights from config path. If missing, log warning and use zero-shot fallback.
    - Return `(clip_model, adapter)`.

**Stage 0: Landmark Crop Extraction** (`landmark_crops.py`):
  - Input: frame `(H,W,3)` + landmarks `(478,2)`
  - Extract **6 anatomical crops**, each targeting a region where forgery artifacts concentrate:
    - `[0] left_periorbital` (landmarks 33,133,160,159,158,144)
    - `[1] right_periorbital` (landmarks 263,362,385,386,387,373)
    - `[2] nasolabial_left` (landmarks 92, 205, 216, 206)
    - `[3] nasolabial_right` (landmarks 322, 425, 436, 426)
    - `[4] hairline_band` (landmarks 10, 338, 297, 332, 284, 103, 67)
    - `[5] chin_jaw` (landmarks 172,136,150,149,176,148,152,377,400,379,365)
  - Per crop: compute bbox from landmarks → pad 20% → clamp to image boundaries → extract at native resolution → resize to 224×224 using `cv2.INTER_LANCZOS4` → apply CLIP normalization (mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`) → output `(1, 3, 224, 224)` tensor.

**Stage 1: Patch Token Extraction** (`patch_extractor.py`):
  - Per crop: forward through frozen CLIP ViT-B/32.
  - Register hooks at `transformer.resblocks` indices **[3, 6, 9, 11]** (out of 12 blocks).
  - Each hook captures `output[1:]` — the **49 patch tokens**, skipping `[CLS]` at index 0.
  - ⚠ **CRITICAL**: ViT-B/32 internal tensor layout is `(seq_len, batch, dim)` NOT `(batch, seq, dim)`.
    - Patch tokens: `output[1:]` → shape `(49, batch, 512)` → after `permute(1,0,2)` → `(batch, 49, 512)` ✓
    - **WRONG (common silent error):** `output[:, 0, :]` — returns wrong data, no error thrown.
  - Hooks **must** be deregistered in a `finally` block. Dangling hooks corrupt all subsequent forward passes.
  - Per crop output: `(1, 4_layers, 49_tokens, 512_dim)`.

**Stage 2a: Spatial Pooling** (`bottleneck.py`):
  - Per crop, per layer: learned weights `(4_layers, 49_tokens)` → softmax over token dim → weighted sum.
  - Transform: `(1, 4, 49, 512)` → `(1, 4, 512)`. Total: 1,176 params.

**Stage 2b: Layer Fusion** (`bottleneck.py`):
  - Per crop: learned layer weights `(4,)` → softmax over layer dim → weighted sum + `LayerNorm(512)`.
  - Transform: `(1, 4, 512)` → `(1, 512)`. Total: 6,168 params.
  - Stack 6 crops → `(1, 6, 512)`.

**Stage 3: Cross-Patch Attention** (`attention_head.py`):
  - Low-rank single-head attention: Q(512→64), K(512→64), V(512→512), Out(512→512).
  - Attention scores: `Q @ K^T / √64` → softmax.
  - **CRITICAL: Diagonal zeroing AFTER softmax**, then renormalize. Zeroing before biases the distribution.
  - Cross-patch attention matrix `(B, 6, 6)` IS the interpretable heatmap.
  - Residual + LayerNorm → `(B, 6, 512)`. Total: ~590K params.
  - **Per-patch scoring**: 6 independent heads: `Linear(512→128) → GELU → Linear(128→1) → Sigmoid` → `(B, 6)`.
  - **LSE Pooling**: `log_beta = nn.Parameter(log(10.0))`, `beta = exp(log_beta)` (always positive).
    `final_score = (max_s + log(Σ exp(beta × scores - max_s))) / beta` → single fake score.
    LSE = smooth max. High beta → approaches max (worst patch dominates). Low beta → approaches mean.

**Stage 4: Test-Time Augmentation** (`tta.py`):
  - Pass 1: original crops → `score_orig`
  - Pass 2: horizontally flipped crops → `score_flip`
  - `final_score = max(score_orig, score_flip)`
  - Patch scores and attention weights always from Pass 1 for consistent explainability.

**Temporal Latent Jitter** (video only, zero additional VRAM):
  - Stream 5 frames through already-loaded adapter bottleneck → compute cosine similarity variance.
  - Generative video (Sora) shows high inter-frame latent variance.
  - `final_clip_score = 0.8 × single_frame_score + 0.2 × jitter_score`

**Heatmap→Text Contract for Phi-3:**
  - Only report patches with score > 0.65. Only report cross-patch attention where off-diagonal weight > 0.25.
  - String ≤ 3 sentences. Example: "CLIP adapter flagged: left_periorbital (0.91), right_periorbital (0.84). Cross-patch attention: left_periorbital → right_periorbital (0.43)."

**Total trainable: ~993K params (~3.8MB). CLIP frozen (~150M params, ~600MB).**

**Section D: Implementation Rules for That Day**
- Import `VRAMLifecycleManager` from `utils.vram_manager`.
- Do not keep the CLIP model instantiated on the class level.

**Section E: Testing & Verification Steps**
Create `test_day11.py`:
```python
import numpy as np
import torch
from core.tools.clip_adapter_tool import ClipAdapterTool
from core.config import AegisConfig

def test_day11():
    tool = ClipAdapterTool()
    
    # Create dummy patches (6 anatomical crops per README)
    patches = {
        "patch_left_periorbital": np.zeros((224, 224, 3), dtype=np.uint8),
        "patch_right_periorbital": np.zeros((224, 224, 3), dtype=np.uint8),
        "patch_nasolabial_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "patch_nasolabial_right": np.zeros((224, 224, 3), dtype=np.uint8),
        "patch_hairline_band": np.zeros((224, 224, 3), dtype=np.uint8),
        "patch_chin_jaw": np.zeros((224, 224, 3), dtype=np.uint8),
    }
    
    result = tool.execute(patches)
    print(f"CLIP Adapter logic evaluated: Score {result.score}, Success: {result.success}")
    assert result.success is True
    assert 0.0 <= result.score <= 1.0
    
    # Verify memory is flushed
    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() == 0
    print("✅ Day 11 CLIP tool tested. VRAM safely relinquished.")

if __name__ == "__main__":
    test_day11()
```
Run `python test_day11.py`.
Expected output: Evaluates dummy data via zero-shot (or adapter if weights supplied), completes without OOM, returns valid `ToolResult`.

**Section F: Files Produced**
- `core/tools/clip_adapter_tool.py`
- Depends on: `clip` library, `VRAMLifecycleManager`.
- Enables: High-confidence detection of fully synthetic diffusions and GANs.

#### Day 11 Summary:
- Files: core/tools/clip_adapter_tool.py
- Depends on: VRAMLifecycleManager
- Enables: Universal forgery detection.

---

### Day 12: Blend Boundary Tool (SBI)

#### Prompt for Day 12:

**Section A: Context Reminder**
Aegis-X is a forensic engine. 
This is Phase 3, Day 12. We are implementing the SBI (Self-Blended Images) detector. It utilizes an EfficientNet-B4 backbone to specifically detect face-swap "blend boundaries". The model is trained to recognize the seam where a source face is pasted onto a target frame, making it generator-agnostic for face-swaps.

**Section B: Today's Objectives**
- Create `core/tools/sbi_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `SBITool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_sbi"`
  - `def _load_model(self) -> torch.nn.Module`:
    - Imports `torchvision.models.efficientnet_b4`.
    - Replaces the classifier head to output a single value (`in_features=1792`, `out_features=1`).
    - Loads weights from `AegisConfig().models.sbi_weights`. (If missing, default to neutral random initialized for testing, but log warning).
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - Expected input: `input_data["face_crop_380"]`, `input_data["landmarks"]`. If missing, return `score=0.0, error=True`.
    - **CRITICAL Conditional Skip**: The agent provides `input_data["clip_score"]`. If `clip_score > 0.70` (from `thresholds.SBI_SKIP_CLIP_THRESHOLD`), the face is fully synthetic → SBI cannot detect it (no blend boundary exists). Return `success=True, score=0.0, confidence=0.0, evidence_summary="Skipped SBI: High CLIP score indicates fully synthetic face (not face-swap). SBI only detects blend boundaries."`. Ensemble receives `(0.0, 0.0)` (abstention).
    - Wrap inference: `with VRAMLifecycleManager(self._load_model) as model:`

    **Dual-Scale Cropping:**
    - Compute face bbox from MediaPipe 478 landmarks (which inherently includes the whole head/forehead). Extract TWO context-expanded crops:
      - **1.15× scale**: `bbox expanded by 15%` → captures tight context around FaceMesh
      - **1.25× scale**: `bbox expanded by 25%` → captures wider context (catches different mask sizes)
      - Both: native crop → Lanczos4 → 380×380 → ImageNet normalize (NOT CLIP normalize)
      - **CRITICAL**: Use `torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0` → ImageNet means `[0.485, 0.456, 0.406]`, stds `[0.229, 0.224, 0.225]`

    **Pass 1: Fast Scoring (torch.no_grad):**
    - `score_115 = model(tensor_1_15x).sigmoid()` — ⚠ Check if checkpoint final layer already includes sigmoid. If `nn.Sigmoid` found: skip `.sigmoid()`. If `nn.Linear`: apply. Double-sigmoid clusters scores near 0.5.
    - `score_125 = model(tensor_1_25x).sigmoid()`
    - `max_score = max(score_115, score_125)`
    - `winning_tensor = tensor with higher score`

    **Conditional GradCAM (Pass 2, only if max_score > 0.60):**
    - If `max_score < SBI_FAKE_THRESHOLD (0.60)` → skip GradCAM, return immediately.
    - Register forward hook on `model._blocks[-1]` (final MBConv block, 448 channels, ~12×12 spatial at B4/380).
    - Clone winning tensor, enable grads.
    - Forward → backward → gradient-weighted activation map:
      `weights = gradients.mean(dim=(2,3))` → `(1, 448)`
      `cam = ReLU(Σ weights × activations)` → `(1, 12, 12)` → normalize to [0,1] → bilinear upsample to 380×380.

    **Region Mapping:**
    - Transform 478 MediaPipe landmarks to 380×380 crop coordinates.
    - Define regions using MediaPipe indices: jaw (`172, 136, 150, 149, 176, 148, 152, 377, 400, 379, 365`), hairline (`10, 338, 297, 332, 284, 103, 67`), cheek (`234, 93, 132, 58, 454, 323, 361, 288`), nose_bridge (`168, 6, 197, 195`).
    - Highest mean CAM value in region → `boundary_region` name.
    - If highest mean < `SBI_GRADCAM_REGION_THRESHOLD (0.40)` → "diffuse" (no clear boundary).

    **Output:**
    `{fake_score, boundary_detected, boundary_region, winning_scale, scores_per_scale: {"1.15x": float, "1.25x": float}, interpretation, compute_ms}`
    - `evidence_summary`: "SBI detector: blend boundary detected at jaw (score: 0.84, scale: 1.25x). Consistent with face-swap compositing artifact." OR "SBI detector: no blend boundary detected (score: 0.31). Does not exclude fully-synthetic generation (Sora, Midjourney)." ← Critical: prevents Phi-3 from treating low SBI as evidence of authenticity.

**Section D: Implementation Rules for That Day**
- Do not forget the conditional skip based on the `clip_score`. This is a critical piece of the 'agent' architecture.
- EfficientNet-B4 expects [380, 380] resolution exactly.

**Section E: Testing & Verification Steps**
Create `test_day12.py`:
```python
import numpy as np
import torch
from core.tools.sbi_tool import SBITool

def test_day12():
    tool = SBITool()
    dummy_input = {"face_crop_380": np.zeros((380, 380, 3), dtype=np.uint8)}
    
    # 1. Test Skip Logic
    skip_input = {**dummy_input, "clip_score": 0.85}
    res_skip = tool.execute(skip_input)
    assert res_skip.score == 0.5
    assert "Skipped SBI" in res_skip.evidence_summary
    print("✅ Day 12 SBI Skip logic works!")
    
    # 2. Test Execution
    run_input = {**dummy_input, "clip_score": 0.4}
    res_run = tool.execute(run_input)
    assert res_run.success is True
    
    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() == 0
    print("✅ Day 12 SBI execution + VRAM flush successful!")

if __name__ == "__main__":
    test_day12()
```
Run `python test_day12.py`.
Expected output: Skip logic bypassed the model successfully, while standard logic evaluated the tensor and purged VRAM.

**Section F: Files Produced**
- `core/tools/sbi_tool.py`
- Depends on: `torchvision`, `VRAMLifecycleManager`
- Enables: Face-swap border detection.

#### Day 12 Summary:
- Files: core/tools/sbi_tool.py
- Depends on: torchvision
- Enables: Blending artifact tracking.

---

### Day 13: Frequency Neural Tool (FreqNet)

#### Prompt for Day 13:

**Section A: Context Reminder**
This is Phase 3, Day 13 of the Aegis-X build. We are implementing the final GPU tool: FreqNet (F3Net ResNet-50). While our DCT tool calculates hard mathematical frequency histograms, FreqNet uses a parallel ResNet architecture to learn abstract frequency-domain anomalies common to GANs.

**Section B: Today's Objectives**
- Create `core/tools/freqnet_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `FreqNetTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_freqnet"`
  **File Structure:** Create a `core/tools/freqnet/` sub-package:
  - `__init__.py`
  - `preprocessor.py` — DCT Conv2d + BT.709 luma (CASE B only)
  - `fad_hook.py` — FAD forward hook + zigzag mask + Z-score
  - `calibration.py` — Load/compute band proportion baseline

  **Entry point:** `core/tools/freqnet_tool.py`:
  - `def _load_model(self) -> torch.nn.Module`:
    - Load the F3Net checkpoint from `AegisConfig().models.freqnet_weights`.
    - **Pre-implementation checkpoint inspection required:** Check if `model.FAD_head` contains `Conv2d(3, 64, kernel_size=8, stride=8, bias=False)`.
      - **CASE A (internal DCT):** yes → model handles DCT internally → pass same ImageNet tensor to both streams.
      - **CASE B (external DCT):** no → we must provide external frequency input via `preprocessor.py`.
    - Load calibration baseline from `calibration/freqnet_fad_baseline.pt` via `calibration.py`.
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - Expected input: `input_data["face_crop_224"]`, `input_data["landmarks"]`.
    - If missing, return `score=0.0, error=True`.
    - Wrap inference: `with VRAMLifecycleManager(self._load_model) as model:`

    **Preprocessing:**
    - Face crop: 1.1× expansion → native crop → Lanczos4 → 224×224 → `to_tensor [0,1]`

    **Stream 1 (Spatial):** ImageNet normalize → `(1, 3, 224, 224)`

    **Stream 2 (Frequency, CASE B — external DCT)** (`preprocessor.py`):
    - BT.709 luma extraction: `Y = 0.2126R + 0.7152G + 0.0722B` → `(1, 1, 224, 224)`
      Implemented via `register_buffer('bt709_luma', ...)` for proper device tracking.
    - Frozen `Conv2d` DCT-II basis: `(64, 1, 8, 8)` stride=8 → `(1, 64, 28, 28)` — 64 frequency coefficients per 8×8 spatial block.
    - Log-compression: `log(|x| + 1)` → `(1, 64, 28, 28)` — compresses dynamic range.
    - Per-channel normalization from calibration set → `(1, 64, 28, 28)`.

    **F3Net Forward (both streams):**
    - Two parallel ResNet-50 branches → FAD module (cross-attention between spatial and frequency features).

    **FAD Hook** (`fad_hook.py`):
    - Register forward hook on FAD module.
    - Hook captures 3 frequency band tensors using JPEG zigzag ordering on 64 DCT coefficients:
      - Base (low freq): coefficients 0-20
      - Mid (mid freq): coefficients 21-41
      - High (high freq): coefficients 42-63
    - Always deregister hook in `finally` block: `handle.remove()`.

    **Z-Score Computation:**
    - For each band b: `E_b = ||activation_b||²`, `P_b = E_b / E_total`
    - `Z_b = (P_b - calibration.mean_b) / calibration.std_b`
    - Anomaly named when `|Z| > FREQNET_Z_THRESHOLD (1.5)`:
      - High-freq anomaly → "GAN texture artifacts or diffusion upscaling"
      - Low-freq anomaly → "global illumination mismatch or face compositing"

    **Classification:** logit → sigmoid → `fake_score`
    ⚠ Same sigmoid check as SBI — verify final layer type.

    **Output:** `{freq_anomaly_score, band_z_scores: {base, mid, high}, anomaly_region, interpretation, compute_ms}`

    **Calibration** (`calibration.py`):
    - One-time script (`scripts/compute_fad_calibration.py`) over 500-1000 FFHQ images → saves `{mean_base, std_base, mean_mid, std_mid, mean_high, std_high}` to `calibration/freqnet_fad_baseline.pt`.

**Section D: Implementation Rules for That Day**
- The high-pass filter is crucial: PyTorch's ResNet must ingest the *residual/high-frequency* elements, not the base RGB.

**Section E: Testing & Verification Steps**
Create `test_day13.py`:
```python
import numpy as np
import torch
from core.tools.freqnet_tool import FreqNetTool

def test_day13():
    tool = FreqNetTool()
    dummy_input = {"face_crop_224": np.ones((224, 224, 3), dtype=np.uint8) * 128}
    
    res = tool.execute(dummy_input)
    assert res.success is True
    
    if torch.cuda.is_available():
        assert torch.cuda.memory_allocated() == 0
    print("✅ Day 13 FreqNet execution + VRAM flush successful!")

if __name__ == "__main__":
    test_day13()
```
Run `python test_day13.py`.
Expected output: FreqNet filter evaluates and returns results without OOM.

**Section F: Files Produced**
- `core/tools/freqnet_tool.py`
- Depends on: `torchvision`, `scipy.fft`
- Enables: Advanced GAN artifact detection.

#### Day 13 Summary:
- Files: core/tools/freqnet_tool.py
- Depends on: torchvision.models
- Enables: Coverage against GAN fingerprints.

---

### Day 14: Tool Registry

#### Prompt for Day 14:

**Section A: Context Reminder**
Aegis-X is an agentic engine. The agent doesn't blindly loop files; it dynamically calls tools by name from a registry.
This is Phase 3, Day 14. All 8 forensic tools are written. Today we group them into a singleton registry for the agent to query and invoke.

**Section B: Today's Objectives**
- Create `core/tools/tool_registry.py` and populate it with instances of all 8 tools.

**Section C: Detailed Specifications**
- `class ToolRegistry`:
  - `def __init__(self)`:
    - Initialize an internal `self.tools: dict[str, BaseForensicTool] = {}`.
    - Import and instantiate all 9 tools (C2PA, RPPG, DCT, Geometry, Illumination, Corneal, ClipAdapter, SBI, FreqNet).
    - Register them by their `tool_name` property.
  - `def execute_tool(self, name: str, input_data: dict) -> ToolResult`:
    - Lookup the tool.
    - If found, call `tool.execute(input_data)`.
    - If not found, return a `ToolResult` marked `success=False`, `error=Tool not found`.
  - `def get_tool_names(self) -> list[str]`: Return a list of all registered tool names.

**Section D: Implementation Rules for That Day**
- Handle potential circular imports gracefully.
- Ensure the registry acts as a factory/singleton so tools don't need to be re-instantiated constantly.

**Section E: Testing & Verification Steps**
Create `test_day14.py`:
```python
from core.tools.tool_registry import ToolRegistry

def test_day14():
    registry = ToolRegistry()
    names = registry.get_tool_names()
    print(f"Registered Tools: {names}")
    
    assert len(names) == 9  # Includes corneal tool
    assert "run_geometry" in names
    assert "check_c2pa" in names
    assert "run_clip_adapter" in names
    
    # Test invalid lookup
    err_res = registry.execute_tool("fake_tool", {})
    assert err_res.success is False
    assert "not found" in err_res.error
    
    print("✅ Day 14 Tool Registry functional and populated with all 8 tools.")

if __name__ == "__main__":
    test_day14()
```
Run `python test_day14.py`.
Expected output: Successfully imports every tool without error and shows a 9-item roster.

**Section F: Files Produced**
- `core/tools/tool_registry.py`
- Depends on: All `core/tools/*` implementations.
- Enables: The agent orchestrator can cleanly request tool executions.

#### Day 14 Summary:
- Files: core/tools/tool_registry.py
- Depends on: All Tool modules
- Enables: Unified point of access for Phase 5 LLM Agent.

---

## PHASE 4: Ensemble, Early Stopping & Memory (Days 15–17)

### Day 15: Weighted Ensemble Scorer

#### Prompt for Day 15:

**Section A: Context Reminder**
Aegis-X uses a weighted ensemble of orthogonal tools to defeat deepfakes. The ensemble score is not a simple average, but rather a confidence-weighted normalized aggregation.
This is Phase 4, Day 15. We are building the mathematics required to aggregate `ToolResult`s into a single 0-1 probability.

**Section B: Today's Objectives**
- Create `utils/ensemble.py`.

**Section C: Detailed Specifications**

Import all constants from `utils/thresholds.py` (weights, discount factors, tool-specific thresholds).

**Core Engineering Invariant:** A tool's voting power in the denominator must always match its informational contribution to the numerator. Every tool returns `(contribution, effective_weight)` — NOT `score × confidence`.

**6 Rules That Prevent Silent Math Bugs:**
1. `_route()` returns `(contribution, effective_weight)` tuple, not float.
2. Denominator uses `effective_weight`, never `base_weight` blindly.
3. All abstentions return `(0.0, 0.0)` — zero contribution, zero weight pull.
4. Discount scales BOTH numerator AND denominator.
5. rPPG uses discrete probability, not continuous score × weight.
6. `PULSE_PRESENT` returns `(0.0, 0.15)` NOT `(-0.15, 0.15)` — no negative probabilities.

- `def _route(tool_result: ToolResult, context: dict) -> tuple[float, float]`:
  Per-tool routing function. `context` contains `dct_double_quant` and `clip_score` from other tools.
  Returns `(contribution, effective_weight)`.

  **Complete Routing Table:**

  | Tool / Case | `contribution` | `effective_weight` |
  |:---|:---|:---|
  | `check_c2pa` (`valid=True`) | *(short-circuit — bypass ensemble entirely)* | — |
  | `check_c2pa` (`valid=False`) | `0.0` | `0.0` |
  | `run_rppg` (`PULSE_PRESENT`) | `0.00` | `0.15` |
  | `run_rppg` (`NO_PULSE`) | `0.15` | `0.15` |
  | `run_rppg` (`AMBIGUOUS`) | `0.0` | `0.0` |
  | `run_dct` (score `s`) | `s × 0.10` | `0.10` |
  | `run_geometry` (fake_score `s`) | `s × 0.03` | `0.03` |
  | `run_illumination` (fake_score `s`) | `s × 0.02` | `0.02` |
  | `run_corneal` (fake_score `s`) | `s × 0.03` | `0.03` |
  | `run_clip_adapter` (fake_score `s`) | `s × 0.30` | `0.30` |
  | `run_sbi` (`score < 0.30`, blind spot) | `0.0` | `0.0` |
  | `run_sbi` (`score > 0.80`, no compression) | `s × 0.20` | `0.20` |
  | `run_sbi` (`score > 0.80`, `dct > 0.70`) | `s × 0.08` | `0.08` |
  | `run_sbi` (mid-band `0.30-0.80`) | `s × eff_w` | `0.03 + (0.12 × clip_score)` |
  | `run_freqnet` (no compression) | `s × 0.20` | `0.20` |
  | `run_freqnet` (`dct > 0.70`) | `s × 0.10` | `0.10` |

  **Cross-tool compression discount:** When `dct_double_quant > DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD (0.70)`, SBI effective weight × `SBI_COMPRESSION_DISCOUNT (0.40)`, FreqNet effective weight × `FREQNET_COMPRESSION_DISCOUNT (0.50)`.

  **Error gating:** If `tool_result.error is True` → return `(0.0, 0.0)` immediately. Never treat errored tools as voting REAL.

- `def calculate_ensemble_score(tool_results: list[ToolResult]) -> dict`:
  - Return: `{"ensemble_score": float, "is_c2pa_override": bool}`.
  - Step 1: C2PA check — if signed, return `ensemble_score=0.0, is_c2pa_override=True`.
  - Step 2: Build context dict `{dct_double_quant, clip_score}` from existing results.
  - Step 3: For each tool result, call `_route(result, context)` → get `(contribution_i, eff_weight_i)`.
  - Step 4: `ensemble_score = sum(contributions) / sum(eff_weights)`. Handle div-by-zero → return 0.5.
  - Step 5: Clamp to [0.0, 1.0].
  - **No "agreement calibration" (std-dev adjustment)**. This does NOT exist in the architecture.

**Section D: Implementation Rules for That Day**
- Ignore tools where `success=False` or `confidence == 0.0` from the normalization calculation completely.

**Section E: Testing & Verification Steps**
Create `test_day15.py`:
```python
from core.data_types import ToolResult
from utils.ensemble import calculate_ensemble_score

def test_day15():
    res1 = ToolResult(tool_name="run_clip_adapter", success=True, score=0.9, confidence=0.9, details={}, execution_time=0.5, evidence_summary="")
    res2 = ToolResult(tool_name="run_geometry", success=True, score=0.8, confidence=0.8, details={}, execution_time=0.1, evidence_summary="")
    
    # 1. Base test
    out = calculate_ensemble_score([res1, res2])
    print(f"Aggregated Score (High agreement): {out['ensemble_score']:.3f}")
    assert out["ensemble_score"] > 0.8
    assert not out["is_c2pa_override"]
    
    # 2. C2PA Override test
    c2pa = ToolResult(tool_name="check_c2pa", success=True, score=0.0, confidence=1.0, details={}, execution_time=0.1, evidence_summary="")
    out_override = calculate_ensemble_score([res1, res2, c2pa])
    print(f"Override Score: {out_override['ensemble_score']:.3f}")
    assert out_override["ensemble_score"] == 0.0
    assert out_override["is_c2pa_override"]
    
    print("✅ Day 15 Ensemble aggregation math passed.")

if __name__ == "__main__":
    test_day15()
```
Run `python test_day15.py`.
Expected output: Evaluates custom weighting rules and respects the cryptographical override.

**Section F: Files Produced**
- `utils/ensemble.py`
- Depends on: `AegisConfig`
- Enables: Final probability resolution.

#### Day 15 Summary:
- Files: utils/ensemble.py
- Depends on: ToolResult
- Enables: Core mathematical decision making.

---

### Day 16: Early Stopping Controller

#### Prompt for Day 16:

**Section A: Context Reminder**
A traditional pipeline executes all steps. Aegis-X runs as an agent and respects an Early Stopping check to save compute time (up to 40-80% savings) on overwhelmingly obvious cases.
This is Phase 4, Day 16. We are building the logic that evaluates if the current ensemble score is locked down enough to skip the remaining heavy GPU tools.

**Section B: Today's Objectives**
- Create `core/early_stopping.py`.

**Section C: Detailed Specifications**
- `class EarlyStoppingController`:
  - `def __init__(self, thresholds: ThresholdConfig)`
  - `def evaluate(self, current_ensemble_score: float, tools_run: list[str], tools_pending: list[str]) -> bool:`
    - Uses threshold bounds (`0.15` and `0.85` by default). Get base weights from `AegisConfig`. Compute `total_base_weights_run = sum(base_weights[t] for t in tools_run)`.
    - Condition 1 (C2PA Signed): If `check_c2pa` was run and `current_ensemble_score == 0.0` (override triggered), STOP = True.
    - Condition 2: If `current_ensemble_score > 0.85` AND `total_base_weights_run > 0.40`, STOP = True. (Locked fake. The 0.40 check prevents catastrophic early stopping if only one tiny 0.03 weight tool has run).
    - Condition 3: If `current_ensemble_score < 0.15` AND `total_base_weights_run > 0.40`, STOP = True. (Locked real).
    - Condition 4 (Diminishing potential): Calculate the theoretical maximum remaining weight. If the pending tools possess too little weight to possibly pull the current score out of the "verdict threshold zone", STOP = True.
  - Return `True` to halt analysis, `False` to continue executing tools.

**Section D: Implementation Rules for That Day**
- Document the diminishing potential formula clearly. To calculate it, use the fixed base weights from `AegisConfig`.

**Section E: Testing & Verification Steps**
Create `test_day16.py`:
```python
from core.early_stopping import EarlyStoppingController
from core.config import AegisConfig

def test_day16():
    esc = EarlyStoppingController(AegisConfig().thresholds)
    
    # Very high score, should trigger early stop
    stop1 = esc.evaluate(0.92, ["run_clip_adapter", "run_freqnet"], ["run_sbi"])
    assert stop1 is True
    
    # Ambiguous score, should NOT stop
    stop2 = esc.evaluate(0.55, ["run_geometry"], ["run_clip_adapter"])
    assert stop2 is False

    print("✅ Day 16 Early Stopping controller obeys threshold rules!")

if __name__ == "__main__":
    test_day16()
```
Run `python test_day16.py`.
Expected output: Boolean results matching expectation.

**Section F: Files Produced**
- `core/early_stopping.py`
- Depends on: `AegisConfig`
- Enables: The agent loop to break early and yield verdicts 2-4x faster.

#### Day 16 Summary:
- Files: core/early_stopping.py
- Depends on: ThresholdConfig
- Enables: Massive compute saving optimization.

---

### Day 17: Local JSON Memory System

#### Prompt for Day 17:

**Section A: Context Reminder**
Aegis-X doesn't just evaluate in a vacuum — it has a persistent memory. If an analyst provides feedback ("This was actually a false positive"), the system stores the vector signature of that file and avoids repeating the mistake.
This is Phase 4, Day 17. We are scaffolding a local JSON-backed memory system for experience learning.

**Section B: Today's Objectives**
- Create `core/memory.py`.

**Section C: Detailed Specifications**
- `class MemorySystem`:
  - `def __init__(self, db_path="data/memory.json")`: 
    Ensure directory `data/` exists. If `db_path` doesn't exist, initialize it with an empty dict `{}` or list `[]`.
    The JSON structure will store a list of cases containing: `id, timestamp, file_hash, file_type, verdict, confidence, ensemble_score, tool_scores (dict), reasoning, feedback_label`.
  - `def store_case(self, file_hash: str, file_type: str, verdict: str, confidence: float, ensemble: float, tool_scores_dict: dict, reasoning: str)`:
    Loads JSON, updates the record for `file_hash` if it exists or appends a new one, and saves it.
  - `def store_feedback(self, file_hash: str, actual_label: str)`:
    Updates `feedback_label` with user correction (e.g., "REAL" or "FAKE") and saves the JSON file.
  - `def query_similar_history(self, current_tool_scores: dict) -> list[dict]`:
    Retrieve past cases from the JSON file. Calculate Euclidean distance between `current_tool_scores` vectors and historical `tool_scores_dict` vectors. Return the top 3 closest historical evaluations where a `feedback_label` exists. This allows the LLM to write: "This exact combination of high CLIP but low FreqNet previously resulted in a False Positive."

**Section D: Implementation Rules for That Day**
- Keep file read/writes atomic if possible. Use Python's built-in `json`.
- The euclidean distance logic is executed by loading the JSON list into memory, calculating distances on `tool_scores_dict` values.

**Section E: Testing & Verification Steps**
Create `test_day17.py`:
```python
import os
from core.memory import MemorySystem

def test_day17():
    os.makedirs("test_data", exist_ok=True)
    db_path = "test_data/test_memory.json"
    if os.path.exists(db_path): os.remove(db_path)
    
    mem = MemorySystem(db_path)
    
    mock_tools = {"run_clip_adapter": 0.9, "run_rppg": 0.2}
    mem.store_case("abc123hash", "image", "FAKE", 0.85, 0.85, mock_tools, "Looks synthetic")
    mem.store_feedback("abc123hash", "REAL") # User corrects to REAL (False Positive scenario)
    
    mock_current = {"run_clip_adapter": 0.88, "run_rppg": 0.25}
    matches = mem.query_similar_history(mock_current)
    
    assert len(matches) == 1
    assert matches[0]["feedback_label"] == "REAL"
    print("✅ Day 17 Local JSON memory stores cases, accepts feedback, and returns euclidean proximity matches!")
    
if __name__ == "__main__":
    test_day17()
```
Run `python test_day17.py`.
Expected output: The JSON creates successfully, accepts operations, retrieves records, and calculates distance accurately.

**Section F: Files Produced**
- `core/memory.py`
- Depends on: `json`
- Enables: State persistence and evolutionary context across multiple runs.

#### Day 17 Summary:
- Files: core/memory.py
- Depends on: json
- Enables: LLM to reference historical fixes.

---
END OF PART 3. Say 'continue' for Part 4: Phase 5 (Days 18–22).
