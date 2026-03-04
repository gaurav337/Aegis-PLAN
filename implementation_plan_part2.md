# AEGIS-X IMPLEMENTATION PLAN: PART 2

## PHASE 2: CPU Forensic Tools (Days 6–10)

### Day 6: C2PA Provenance Tool

#### Prompt for Day 6:

**Section A: Context Reminder**
You are building Aegis-X, an agentic deepfake detection system. We have completed the base infrastructure, configuration, preprocessing (**MediaPipe 478-point** crops), and `BaseForensicTool`.
This is Phase 2, Day 6. We are building the first of the 5 CPU-only tools: The C2PA Provenance Tool. It's unique because it returns a binary verification rather than a floating-point anomaly score.

**Section B: Today's Objectives**
- Create `core/tools/c2pa_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `C2PATool(BaseForensicTool)`:
  - `@property tool_name`: return `"check_c2pa"`
  - `def setup(self)`: Import verification for `c2pa-python`.
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - `input_data` MUST contain `"media_path"` (a `Path` or `str` to the original image/video).
    - Use the `c2pa.read_file(media_path)` logic (from the `c2pa-python` library).
    - If valid signatures are found, extract the `signer` and `timestamp`.
    - Returns a `ToolResult`:
      - If signed: `success=True`, `score=0.0` (0% fake, it's verified real), `confidence=1.0`.
      - If NOT signed (or no C2PA data exists): `success=True`, `score=0.0`, `confidence=0.0`. Returns `(0.0, 0.0)` in ensemble — zero contribution, zero weight pull. Does NOT vote REAL or FAKE. Absence of C2PA is not evidence of anything.
      - `evidence_summary`: "Signed by [Signer] at [Timestamp]" OR "No C2PA provenance data found."
      - `details`: Dict containing raw extracted C2PA metadata if available.

**Section D: Implementation Rules for That Day**
- Handle the case where `c2pa.read_file` raises an exception (e.g., file not supported or missing library) — catch it natively within `_run_inference` and return the "neutral" `score=0.5` result rather than crashing. 

**Section E: Testing & Verification Steps**
Create `test_day6.py`:
```python
import tempfile
from pathlib import Path
from core.tools.c2pa_tool import C2PATool

def test_day6():
    tool = C2PATool()
    tool.setup()
    
    # Create an empty file to test the 'unsigned' logic
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        mock_path = Path(f.name)
        
    try:
        result = tool.execute({"media_path": mock_path})
        print(f"C2PA Result on unsigned file: {result}")
        assert result.success is True
        assert result.score == 0.0  # Abstention: (0.0, 0.0) in ensemble
        assert result.confidence == 0.0
        assert "No C2PA" in result.evidence_summary
        print("✅ Day 6 C2PA tool tested successfully! Neutral fallback works.")
    finally:
        mock_path.unlink()

if __name__ == "__main__":
    test_day6()
```
Run `python test_day6.py`.
Expected output: The tool gracefully returns `score=0.0` with 0 confidence (ensemble abstention) for an unsigned/empty image file.

**Section F: Files Produced**
- `core/tools/c2pa_tool.py`
- Depends on: `c2pa-python`, `BaseForensicTool`.
- Enables: The agent can rapidly clear content-credentialed images.

#### Day 6 Summary:
- Files: core/tools/c2pa_tool.py
- Depends on: c2pa-python
- Enables: Fast-path exiting for cryptographically signed media.

---

### Day 7: Biological Liveness Tool (rPPG)

#### Prompt for Day 7:

**Section A: Context Reminder**
Aegis-X uses physics and biological signals to defeat deepfakes. 
This is Phase 2, Day 7. Today we implement the POS (Plane Orthogonal to Skin-tone) rPPG algorithm. Generated faces lack a biological pulse. You are adapting the mathematical equations strictly from the Aegis-X README into `run_rppg`. **NEVER return a BPM.** Only return liveness confidence.

**Section B: Today's Objectives**
- Create `core/tools/rppg_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `RPPGTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_rppg"`
  - `def setup(self)`: No-op.
   - `def _extract_forehead_roi(self, frame: np.ndarray, current_bbox: tuple, relative_forehead_box: tuple) -> np.ndarray`:
     - Apply the `relative_forehead_box` proportional scalars (x_min%, y_min%, x_max%, y_max%) to the `current_bbox` `[x1, y1, x2, y2]`. Clamp to image boundaries. Return the cropped ROI.
     - **CRITICAL — Hair Occlusion Guardrail:** Subjects with bangs (hair covering forehead) produce noisy high-variance signals that corrupt the POS algorithm. Before running POS, on the first frame's ROI, compute the **RGB variance**. If `np.std(roi_pixels) > 35.0`, the ROI is hair-contaminated — return `"AMBIGUOUS"` immediately.
   - `def _run_inference(self, input_data: dict) -> ToolResult`:
     - `input_data` MUST contain `"frames_30fps"` AND `"tracked_faces"`. Iterate over each `face` in `tracked_faces`. Ensure `face["trajectory_bboxes"]` exists.
     - Requirements: `min_frames = 90` (from `utils/thresholds.py`). If `<90`, return neutral result (`score=0.5, confidence=0.0`).
     - **Step 0: Anchor Geometry** — Read the sharpest frame's `face["landmarks"]`. Compute the absolute bounding box of the MediaPipe upper hairline band (nodes `10, 338, 297, 332, 284, 103, 67`). Compute this box's proportional boundaries `relative_forehead_box` relative to the *overall face bounding box* for that specific anchored frame.
     - **Step 1: Extract temporal RGB signal**
       For `f_idx, frame` in enumerate `frames_30fps`:
         Get `curr_box = face["trajectory_bboxes"][f_idx]`.
         Extract the dynamically tracked ROI using `_extract_forehead_roi(frame, curr_box, relative_forehead_box)`.
         (If `f_idx == 0` and it returns `"AMBIGUOUS"`, break and return `score=0.5, confidence=0.0, evidence_summary="Ambiguous: hair occlusion."`)
         Compute spatial mean RGB values -> `(N, 3)` matrix.
     - **Step 2: POS (Plane Orthogonal to Skin-tone) Algorithm**
       Sliding window of 1.6 seconds (48 frames at 30fps):
       - Normalize: `Cn = RGB[m:n] / mean(RGB[m:n])`
       - POS projection: `S = [[0, 1, -1], [-2, 1, 1]] @ Cn.T`
       - Extract pulse: `h = S[0] + (std(S[0]) / std(S[1])) * S[1]`
       - Normalize and accumulate into BVP signal H
     - **Step 3: Frequency analysis**
       FFT periodogram (`nfft = max(2048, next_power_of_2(len(signal)))`)
       Band-limit to 0.7-2.5 Hz (42-150 BPM cardiac range)
       Find peak frequency -> compute SNR:
       `signal_power = sum(PSD within +/-0.1 Hz of peak)`
       `noise_power = sum(PSD outside pulse band)`
       `SNR_dB = 10 * log10(signal_power / noise_power)`
     - **Step 4: Three-tier composite scoring** (NOT binary)
       ```
       score = 0.0
       if SNR > 3.0 dB (clean signal):       score += 0.4
       if 40 <= HR_BPM <= 150 (physiological): score += 0.3
       if HR_std < 8 BPM (stable):             score += 0.3
       ```
     - **Step 5: Three-tier discrete output for ensemble**
       - `PULSE_PRESENT`: score >= 0.7 -> ensemble `(0.00, 0.15)` <- real signal, dampens fake probability
       - `NO_PULSE`: score <= 0.3 -> ensemble `(0.15, 0.15)` <- strong fake signal
       - `AMBIGUOUS`: otherwise -> ensemble `(0.0, 0.0)` <- abstains entirely
     - **NEVER** return BPM. Reports liveness confidence only.
     - `evidence_summary`: "Physiological pulse signal detected (SNR: X.X dB). Liveness confirmed." OR "No biological pulse detected (flatline). Consistent with synthetic media." OR "Ambiguous pulse signal. Insufficient evidence for liveness determination."

**Section D: Implementation Rules for That Day**
- Strictly avoid calculating exact BPM integers, focus only on the Power Ratio.
- DO NOT return or log the BPM. The UI explicitly forbids showing corrupted BPMs; it only cares about the *presence* of the biological signal.

**Section E: Testing & Verification Steps**
Create `test_day7.py`:
```python
import numpy as np
from core.tools.rppg_tool import RPPGTool

def test_day7():
    tool = RPPGTool()
    
    def test_day7_rppg():
        tool = RPPGTool()
        tool.setup()
        
        # 90 frames of random noise
        fake_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(90)]
        dummy_trajectory = {i: (100, 100, 300, 300) for i in range(90)}
        dummy_tracked_faces = [{"identity_id": 0, "landmarks": np.zeros((478, 2)), "trajectory_bboxes": dummy_trajectory}] 
        
        result = tool.execute({"frames_30fps": fake_frames, "tracked_faces": dummy_tracked_faces})
        
        assert result.success is True
        # Noise should not contain a biological pulse
        assert result.score > 0.8  # Should flag as fake due to flatline
        print(f"rPPG Output: Score={result.score}, Conf={result.confidence}")
        print(f"Summary: {result.evidence_summary}")
        print("✅ Day 7 rPPG tool correctly identified flatline synthetic data!")
        
    test_day7_rppg()

if __name__ == "__main__":
    test_day7()
```
Run `python test_day7.py`.
Expected output: The POS algorithm processes the random noise, detects no periodic pulse (flatline), and assigns a high fake score.

**Section F: Files Produced**
- `core/tools/rppg_tool.py`
- Depends on: `scipy`, `numpy`
- Enables: Liveness detection for video inputs.

#### Day 7 Summary:
- Files: core/tools/rppg_tool.py
- Depends on: scipy, numpy
- Enables: Real-time pulse extraction from webcam/video footage.

---

### Day 8: DCT Frequency Analysis Tool

#### Prompt for Day 8:

**Section A: Context Reminder**
We are building Aegis-X. Deepfakes undergo multiple compression cycles on social media.
This is Phase 2, Day 8. We are building the DCT Frequency Analysis Tool. By analyzing the 8x8 block Discrete Cosine Transform (DCT) histogram, we can detect double-quantization signals characteristic of re-compressed deepfakes.

**Section B: Today's Objectives**
- Create `core/tools/dct_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `DCTTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_dct"`
    - `def _run_inference(self, input_data: dict) -> ToolResult`:
    1. Read `input_data["tracked_faces"]` (list of dictionaries). Ensure each contains a `face_crop_224` numpy array. If list is missing or empty, return abstention.
    2. Iterate over each `face` in `tracked_faces`. Get the `face_crop_224`.
    3. Convert the RGB crop to grayscale using `cv2.cvtColor`.
    3. Ensure dimensions are divisible by 8 (pad or crop).
    4. Group into 8x8 blocks. Reshape the entire array: `blocks = image.reshape(h//8, 8, w//8, 8).transpose(0, 2, 1, 3)`.
    5. Apply 2D DCT on every block: `scipy.fft.dctn` (with `norm='ortho'`). DO NOT use 1D `scipy.fftpack.dct`.
    6. Flatten the AC coefficients (ignore the [0,0] DC component of each 8x8 block).
    7. Compute the histogram of these coefficients. Find periodic peaks via autocorrelation:
       `autocorr = np.correlate(hist, hist, mode='same')`
    8. Measure double quantization: The amplitude of secondary peaks relative to the primary peak. If `peak_ratio > 0.15`, flag.
    9. Calculate `score`: `min(1.0, peak_ratio / 0.30)`.
    10. `confidence`: `min(0.9, score + 0.2)`
    11. `evidence_summary`: "DCT analysis detected double-quantization artifacts indicating structural modification" or "Smooth DCT frequency distribution consistent with natural imagery."

**Section D: Implementation Rules for That Day**
- Strictly use `scipy.fft.dctn` for 2D transform processing.
- This is a mathematical transformation, handle zeros or `NaNs` safely (add `1e-10` to denominators).

**Section E: Testing & Verification Steps**
Create `test_day8.py`:
```python
import numpy as np
import cv2
from core.tools.dct_tool import DCTTool

def test_day8():
    tool = DCTTool()
    
    # Generate a pure white image (no texture, no double quant expected)
    clean_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
    
    # Generate an image with harsh 8x8 blocking artifacts
    blocky_img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            blocky_img[i:i+8, j:j+8] = 255
            
    # Use dummy face format
    dummy_tracked_faces_clean = [{"identity_id": 0, "face_crop_224": clean_image}]
    dummy_tracked_faces_blocky = [{"identity_id": 0, "face_crop_224": blocky_img}]

    res_clean = tool.execute({"tracked_faces": dummy_tracked_faces_clean})
    res_blocky = tool.execute({"tracked_faces": dummy_tracked_faces_blocky})
    
    print(f"Clean Score: {res_clean.score}, Blocky Score: {res_blocky.score}")
    assert res_clean.success and res_blocky.success
    # Blocky image should yield a higher artifact score than the perfectly smooth one
    assert res_blocky.score > res_clean.score
    print("✅ Day 8 DCT tool tested. Differentiates smooth vs high-frequency grid artifacts.")

if __name__ == "__main__":
    test_day8()
```
Run `python test_day8.py`.
Expected output: Success message confirming the algorithm differentiates smooth vs hard-gradient images.

**Section F: Files Produced**
- `core/tools/dct_tool.py`
- Depends on: `scipy.fft`.
- Enables: Analysis of compression-surviving artifacts.

#### Day 8 Summary:
- Files: core/tools/dct_tool.py
- Depends on: scipy.fft
- Enables: Detection of compression and quantization artifacts.

---

### Day 9: Geometric Physics Tool

#### Prompt for Day 9:

**Section A: Context Reminder**
Aegis-X checks whether facial structures adhere to anatomical physics. Generative models hallucinate visual textures but often fail basic 3D human constraints.
This is Phase 2, Day 9. We are implementing 7 explicit anthropometric measurements on the extracted **MediaPipe 478-point** landmarks. This tool is a cornerstone of our zero-shot physics defense.

**Section B: Today's Objectives**
- Create `core/tools/geometry_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `GeometryTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_geometry"`
  - `def _run_inference(self, input_data: dict) -> ToolResult`:
    - Reads `tracked_faces` from `input_data`. Iterate over each `face`. Extract `landmarks` (np.ndarray of shape **[478, 2]**, MediaPipe pixel coordinates). If missing, return neutral result `score=0.5`.
    - Loop logic: compute metrics mapped to the highest scoring outlier among tracked identities.
    - Function: `def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))`
    - You MUST implement EXACTLY these **7 distinct checks** using `dist` with **MediaPipe landmark indices**:
      1. **IPD ratio**: `dist(landmarks[33], landmarks[263]) / dist(landmarks[234], landmarks[454])`. Uses MediaPipe outer iris centers (nodes 33/263), divided by jaw-width (234/454). Valid: 0.42 to 0.52.
      2. **Philtrum ratio**: `dist(landmarks[94], landmarks[0]) / dist(landmarks[168], landmarks[152])`. MediaPipe: columella base (94), lip midpoint (0), nose bridge (168), chin (152). Valid: 0.10 to 0.15.
      3. **Eye width asymmetry**: `abs(dist(lm[33], lm[133]) - dist(lm[263], lm[362])) / dist(lm[234], lm[454])`. MediaPipe left eye outer/inner corners vs right eye. Flag if > 0.05.
      4. **Jaw yaw symmetry (Pose Gate)**: `yaw_proxy = abs(eye_mid_x - landmarks[1].x) / face_width` where `eye_mid_x = (landmarks[33].x + landmarks[263].x) / 2`, `face_width = dist(landmarks[234], landmarks[454])`.
         - **CRITICAL**: If `yaw_proxy > GEOMETRY_YAW_SKIP_THRESHOLD (0.18)`, skip bilateral Checks 3, 4, 5, and 6 (face is too profiled for 2D bilateral measurements to be reliable). Checks 1, 2, and 7 always run.
      5. **Nose width ratio**: `dist(lm[98], lm[327]) / IPD`. MediaPipe alar base nodes 98 (left), 327 (right). Valid: 0.55 to 0.70.
      6. **Mouth width ratio**: `dist(lm[61], lm[291]) / IPD`. MediaPipe mouth corner nodes 61 (left), 291 (right). Valid: 0.85 to 1.05.
      7. **Vertical thirds**: `upper = lm[10] to lm[168]` (hairline to nose bridge), `mid = lm[168] to lm[94]` (nose bridge to columella), `lower = lm[94] to lm[152]` (columella to chin). Check if any third deviates by > 15% from the mean third.
    - Each check that fails adds a string to a `violations` list.
    - `fake_score = len(violations) / 7.0`
    - `confidence = 0.8` (if landmarks exist).
    - `evidence_summary`: Include strings of EXACTLY which anatomical structures are violating constraints (e.g., "Violations found in Jaw yaw symmetry and Philtrum ratio.").

**Section D: Implementation Rules for That Day**
- Use precise indexing on the `[478, 2]` MediaPipe array.
- Import `GEOMETRY_YAW_SKIP_THRESHOLD` from `utils/thresholds.py` — do NOT hardcode `0.18`. This is the single source of truth.
- Catch `ZeroDivisionError` natively with `+ 1e-10` on all denominators.

**Section E: Testing & Verification Steps**
Create `test_day9.py`:
```python
import numpy as np
from core.tools.geometry_tool import GeometryTool

def test_day9():
    tool = GeometryTool()
    
    # MediaPipe 478-point layout (approximate positions for testing)
    landmarks = np.zeros((478, 2))
    # Face width: jaw nodes 234 (left ear) and 454 (right ear)
    landmarks[234] = [0, 50]      # left jaw (ear level)
    landmarks[454] = [100, 50]    # right jaw (ear level)
    # Eye outer corners
    landmarks[33]  = [25, 25]     # left eye outer
    landmarks[133] = [40, 25]     # left eye inner
    landmarks[263] = [60, 25]     # right eye inner
    landmarks[362] = [75, 25]     # right eye outer
    # Nose tip and structural landmarks
    landmarks[1]   = [50, 50]     # nose tip (yaw probe)
    landmarks[168] = [50, 15]     # nose bridge (for vertical thirds upper)
    landmarks[94]  = [50, 60]     # columella base
    landmarks[98]  = [40, 62]     # left alar base (nose width)
    landmarks[327] = [60, 62]     # right alar base (nose width)
    # Mouth corners
    landmarks[61]  = [30, 70]     # mouth left
    landmarks[291] = [70, 70]     # mouth right
    landmarks[152] = [50, 100]    # chin
    landmarks[10]  = [50, 5]      # hairline top
    
    dummy_tracked_faces = [{"identity_id": 0, "landmarks": landmarks}]
    result = tool.execute({"tracked_faces": dummy_tracked_faces})
    print(f"Geometry Score: {result.score}")
    assert result.success is True
    assert "violations" in result.details
    print("✅ Day 9 Geometry tool evaluated layout array successfully!")

if __name__ == "__main__":
    test_day9()
```
Run `python test_day9.py`.
Expected output: The algorithm parses the synthetic array, calculates ratios, determines what passed/failed, and outputs a score without IndexErrors.

**Section F: Files Produced**
- `core/tools/geometry_tool.py`
- Depends on: valid **[478, 2]** MediaPipe landmarks from Preprocessor.
- Enables: High success rate against AI generators that fail 3D anatomical layout.

#### Day 9 Summary:
- Files: core/tools/geometry_tool.py
- Depends on: Preprocessor (MediaPipe 478-point landmarks)
- Enables: Zero-shot anatomical physics verification.

---

### Day 10: Illumination Physics Tool

#### Prompt for Day 10:

**Section A: Context Reminder**
Diffusion models generate photorealistic faces, but they often composite them into scenes with conflicting directional light sources. 
This is Phase 2, Day 10. We are using Horn's "Shape-from-Shading" logic to compare the dominant illumination gradient of the face against the immediate background.

**Section B: Today's Objectives**
- Create `core/tools/illumination_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `IlluminationTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_illumination"`
    - `def _run_inference(self, input_data: dict) -> ToolResult`:
    1. Read `input_data["tracked_faces"]`. For each `face`, read `face_crop_224` and `input_data["media_path"]`. If either missing, return neutral score.
    2. Verify `run_corneal()` ran previously and didn't fail.
    3. Convert RGB crop to `YCrCb` and isolate the `Y` (Luma) channel.
    4. `midpoint_x = 112` -> Since `face_crop_224` is centered and size 224, strictly slice down the middle to avoid misalignments with global coordinate spaces.
    5. Calculate Face Lighting: Mean Luma of the Left side of the face (`image[:, :midpoint_x]`) vs Mean Luma of the Right side of the face (`image[:, midpoint_x:]`).
       `face_grad = abs(face_l - face_r) / (face_l + face_r + 1e-6)`.
    6. Interpret diffuse lighting: If `face_grad < 0.05`, the lighting is diffuse (straight-on flat lighting). Return `score=0.2` (typically real).
     7. Calculate Scene Context: Extract the **bottom 20 rows** of the face crop as the context/neck region: `ctx_l = Y[-20:, :midpoint_x].mean()`, `ctx_r = Y[-20:, midpoint_x:].mean()`. Self-contained on the face crop -- no need to load the full frame.
    8. Which is brighter?
       `face_dom = "left" if face_l > face_r else "right"`
       `ctx_dom = "left" if ctx_l > ctx_r else "right"`
    9. Calculate mismatch penalty:
       - If `face_dom == ctx_dom`: `fake_score = face_grad * 0.20`
       - If mismatch (`!=`): `fake_score = 0.30 + (face_grad * 0.70)`
    10. `confidence`: `min(0.9, face_grad * 10)` (High gradient = strong lighting = confident result).
    11. `evidence_summary`: "Face illumination direction mismatches environmental scene context" or "Consistent lighting found."

**Section D: Implementation Rules for That Day**
- Use standard `numpy` slicing to divide the face. Wait to do this until converting the image to grayscale/luma.
- Always include `1e-6` on denominators.

**Section E: Testing & Verification Steps**
Create `test_day10.py`:
```python
import numpy as np
from core.tools.illumination_tool import IlluminationTool

def test_day10():
    tool = IlluminationTool()
    
    # Create a dummy image 224x224, 3 channel
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Bright left face
    dummy_img[50:150, 50:100] = 200
    # Dark right face
    dummy_img[50:150, 100:150] = 50
    
    # Context (neck area) MISMATCH: Bright right, dark left
    dummy_img[200:220, 50:100] = 50
    dummy_img[200:220, 100:150] = 200
    
    dummy_landmarks = np.zeros((68, 2))
    dummy_landmarks[27:34, 0] = 100  # Sets the midline X to 100
    
    dummy_tracked_faces = [{
        "identity_id": 0,
        "face_crop_224": dummy_img,
        "landmarks": dummy_landmarks
    }]
    result = tool.execute({"tracked_faces": dummy_tracked_faces, "media_path": "fake.mp4"})
    print(f"Illumination Mismatch Score: {result.score}")
    assert result.success is True
    # The intentional mismatch should trigger a fake_score > 0.30
    assert result.score > 0.30
    print("✅ Day 10 Illumination tool caught the synthetic face-scene mismatch!")

if __name__ == "__main__":
    test_day10()
```
Run `python test_day10.py`.
Expected output: The algorithm processes the starkly mismatched array and produces a high fake penalty score.

**Section F: Files Produced**
- `core/tools/illumination_tool.py`
- Depends on: cv2 (for YCrCb conversion)
- Enables: Last of the 5 CPU-bound, instant forensic tools.

#### Day 10 Summary:
- Files: core/tools/illumination_tool.py
- Depends on: cv2, numpy
- Enables: Mismatch detection against Diffusion models.


---

### Day 10.5: Corneal Reflection Consistency Tool

#### Prompt for Day 10.5:

**Section A: Context Reminder**
Diffusion models (Midjourney, DALL-E, Stable Diffusion) consistently fail to synthesize physically consistent specular highlights (catchlights) in both eyes simultaneously. Real eyes reflect the same light sources from symmetric positions.
This is Phase 2, Day 10.5. We are building a physics-based corneal reflection consistency checker.

**Section B: Today's Objectives**
- Create `core/tools/corneal_tool.py`, extending `BaseForensicTool`.

**Section C: Detailed Specifications**
- `CornealTool(BaseForensicTool)`:
  - `@property tool_name`: return `"run_corneal"`
    - `def _run_inference(self, input_data: dict) -> ToolResult`:
    1. Read `input_data["tracked_faces"]`. Iterate through `face`. Find `face_crop_224` and `landmarks`. If either is None, return abstention (`score=0.0, error=True`).
    2. Extract **iris center regions** using **MediaPipe iris nodes 468 (left iris center) and 473 (right iris center)**. Each iris node gives a pixel coordinate. Extract a **15×15 pixel box** centered on each iris node from the native-resolution face crop (scaled to crop coordinates first).
       - Do NOT use eye boundary points; these are not corneal center trackable positions.
    3. Within each 15×15 box, threshold for specular highlights (brightest pixels, top 2%).
    4. If no catchlight detected in either iris region → return `error=True` (abstention, weight 0.0 in ensemble).
    5. Compute spatial centroid offsets for left and right iris catchlights.
    6. Apply mirror-axis correction (left/right iris reflection rays are symmetric about nose node 1).
    7. Measure divergence: `divergence = ||left_offset - right_offset||`.
    8. `fake_score = min(1.0, divergence / max_allowable_divergence)`.
    9. `consistent = fake_score < 0.5`.
    10. `confidence = 0.7` when catchlights detected.
    11. `evidence_summary`: "Corneal reflections consistent between both eyes." OR "Asymmetric corneal reflections detected — specular highlights diverge between left and right eyes."

**Section D: Implementation Rules for That Day**
- This tool has very low ensemble weight (0.03) — it's a supplementary signal.
- Handle cases where eyes are closed or occluded by returning abstention `error=True`.

**Section E: Testing & Verification Steps**
Create `test_day10_5.py`:
```python
import numpy as np
from core.tools.corneal_tool import CornealTool

def test_day10_5():
    tool = CornealTool()
    
    # Create a dummy image with no bright spots (should abstain)
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 80
    dummy_landmarks = np.zeros((68, 2))
    dummy_landmarks[36:42] = [[40, 40], [45, 38], [50, 40], [45, 42], [40, 42], [42, 40]]
    dummy_landmarks[42:48] = [[60, 40], [65, 38], [70, 40], [65, 42], [60, 42], [62, 40]]
    
    dummy_tracked_faces = [{
        "identity_id": 0,
        "face_crop_224": dummy_img,
        "landmarks": dummy_landmarks
    }]
    result = tool.execute({"tracked_faces": dummy_tracked_faces})
    assert result.success  # Should not crash
    print(f"Corneal Score: {result.score}")
    print("✅ Day 10.5 Corneal tool executed without crash!")

if __name__ == "__main__":
    test_day10_5()
```
Run `python test_day10_5.py`.
Expected output: Tool returns without crashing, likely abstaining (no catchlights in uniform gray image).

**Section F: Files Produced**
- `core/tools/corneal_tool.py`
- Depends on: cv2, numpy, **MediaPipe iris nodes 468/473** (requires `refine_landmarks=True` in Preprocessor)
- Enables: Physics-based catch for diffusion model eye artifacts.

#### Day 10.5 Summary:
- Files: core/tools/corneal_tool.py
- Depends on: cv2, numpy
- Enables: Supplementary physics signal for ensemble.
---
END OF PART 2. Say 'continue' for Part 3: Phase 3 (Days 11–14).
