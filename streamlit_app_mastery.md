# `main.py`, `app.py` & Streamlit Mastery Guide (L3 Engineer Depth)

This guide covers:
1. **`main.py`** — Exact purpose, line-by-line breakdown, and Python execution internals.
2. **`app.py` (Lines 1–79)** — Line-by-line, function-by-function, parameter-by-parameter walkthrough including Pillow & NumPy image-processing math.
3. **Streamlit Architecture & Mastery** — Everything you need to understand and defend Streamlit at an L3 engineering level.

---

## PART 1: `main.py` — Line-by-Line & Internals

```python
def main():
    print("Hello from ml-mini!")


if __name__ == "__main__":
    main()
```

### Where did `main.py` come from?
When we initialized the project with `uv init --no-readme`, `uv` automatically scaffolded a standard Python project skeleton (`pyproject.toml`, `.python-version`, `.gitignore`, and `main.py`). It is **not** used by the notebook or Streamlit app—our actual ML training lives in `digit_recognition.ipynb` and our web app lives in `app.py`.

### Line-by-Line Mechanics
- **Lines 1–2 (`def main(): ...`):** Defines a zero-argument entry function encapsulating startup logic instead of putting executable statements at module top-level.
- **Line 5 (`if __name__ == "__main__":`):**
  - Every Python module has a built-in dunder attribute `__name__`.
  - When you run `uv run python main.py` directly, Python sets `__name__ = "__main__"`, so `main()` executes.
  - When another file runs `import main`, Python sets `__name__ = "main"`, so `main()` does **not** run automatically as a side effect of importing.
  - **Why this matters in ML engineering:** Wrapping training or CLI code in `if __name__ == "__main__":` is also mandatory on macOS/Windows when using Python `multiprocessing` (like `joblib` / `n_jobs=-1` in `KNeighborsClassifier`), because spawned worker processes re-import the script—without that guard, workers recursively spawn infinite processes!

---

## PART 2: `app.py` — Line-by-Line, Parameter-by-Parameter Breakdown

### Lines 1–7: Imports & Page Configuration
```python
import streamlit as st
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Handwritten Digit Recognition")
st.title("Handwritten Digit Recognition")
```
- **`import streamlit as st`**: Standard alias for Streamlit's declarative UI library.
- **`from PIL import Image`**: Imports the `Image` module from **Pillow** (fork of PIL—Python Imaging Library) for decoding PNG/JPG bytes, grayscale conversion, cropping, and high-quality resampling.
- **`st.set_page_config(page_title="Handwritten Digit Recognition")`**:
  - **Critical Rule:** Must be the **very first Streamlit command** called in the script (and called only once), otherwise Streamlit raises `StreamlitSetPageConfigMustBeFirstCommandError` because it sets the HTML `<head>` title, favicon, and layout before rendering body elements.
  - **Full parameter list:**
    - `page_title`: Browser tab title (`str`).
    - `page_icon`: Emoji or image path for browser favicon (default `None` → Streamlit crown).
    - `layout`: `"centered"` (default, fixed max-width ~730px) or `"wide"` (uses full viewport width, great for dashboards).
    - `initial_sidebar_state`: `"auto"`, `"expanded"`, or `"collapsed"`.
- **`st.title(...)`**: Renders an `<h1>` heading element in Markdown/DOM.

---

### Lines 9–22: Model Selection & Artifact Loading
```python
dataset_choice = st.selectbox("Select Model:", ["MNIST (28x28)", "load_digits (8x8)"])

samples = np.load("sample_digits.npz")
if dataset_choice == "MNIST (28x28)":
    model = joblib.load("mnist_model.pkl")
    sample_images = samples["mnist_images"]
    sample_labels = samples["mnist_labels"]
    img_size = 28
else:
    model = joblib.load("digits_model.pkl")
    sample_images = samples["digits_images"]
    sample_labels = samples["digits_labels"]
    img_size = 8
```
- **`st.selectbox(label, options, index=0, ...)`**:
  - Renders a dropdown widget. Returns the currently selected item (`str`) on every script rerun.
  - `index=0` (default) selects `"MNIST (28x28)"` on initial page load.
  - Other useful parameters: `key=` (binds to `st.session_state`), `help=` (tooltip), `disabled=False`, `on_change=` (callback).
- **`samples = np.load("sample_digits.npz")`**:
  - `.npz` is a **ZIP archive** of multiple `.npy` binary arrays (`digits_images`, `digits_labels`, `mnist_images`, `mnist_labels`).
  - `np.load` returns an `NpzFile` object that **lazily** reads and decompresses individual arrays only when indexed via `samples["mnist_images"]`.
- **`joblib.load(...)`**:
  - Deserializes the saved `MLPClassifier` instance (`mnist_model.pkl` with 784 inputs / 101,770 params, or `digits_model.pkl` with 64 inputs / 4,810 params) back into memory with all learned `coefs_` and `intercepts_` restored.

---

### Lines 24–39: Input Mode 1 — Selecting a Test Sample Digit
```python
mode = st.radio("Choose input method:", ["Select a sample digit", "Upload an image"])

if mode == "Select a sample digit":
    idx = st.slider("Pick sample index", 0, len(sample_images) - 1, 0)
    img = sample_images[idx]
    true_label = sample_labels[idx]

    st.image(img, caption=f"Actual digit: {true_label}", width=150, clamp=True)

    x_input = img.flatten().reshape(1, -1)
    probs = model.predict_proba(x_input)[0]
    pred = int(np.argmax(probs))
    conf = probs[pred] * 100

    st.subheader(f"Predicted: {pred} ({conf:.2f}% confidence)")
    st.bar_chart(probs)
```
- **`st.radio(label, options, index=0, horizontal=False)`**: Mutually exclusive radio buttons returning the chosen string.
- **`st.slider(label, min_value=0, max_value=99, value=0, step=1)`**:
  - Returns an integer index `0..99`. Every time you drag and release the slider, Streamlit re-executes `app.py` from Line 1 with the new `idx`.
- **`st.image(img, caption=..., width=150, clamp=True)`**:
  - `img` has shape `(28, 28)` or `(8, 8)` with `float32`/`float64` values in `[0.0, 1.0]`.
  - **`width=150`**: Upsamples display to 150px wide in the browser so tiny 8×8 or 28×28 images are clearly visible.
  - **`clamp=True`**: **Crucial!** Clamps floating-point pixel values strictly into `[0.0, 1.0]` before PNG encoding—without `clamp=True`, floating-point rounding like `1.0000001` or `-1e-8` raises a `ValueError` in Streamlit!
- **`x_input = img.flatten().reshape(1, -1)`**:
  - `.flatten()` turns `(28, 28)` → `(784,)` (1D copy).
  - `.reshape(1, -1)` turns `(784,)` → `(1, 784)` (2D batch of 1 sample; `-1` tells NumPy to infer the column count automatically). `scikit-learn` strictly requires 2D input `(n_samples, n_features)`.
- **`probs = model.predict_proba(x_input)[0]`**:
  - Runs the MLP forward pass (`ReLU` hidden layer → `Softmax` output layer), returning shape `(1, 10)`. Indexing `[0]` extracts the 1D array of 10 probabilities summing to `1.0`.
- **`pred = int(np.argmax(probs))`**:
  - `np.argmax` returns the index (`0–9`) with highest probability as `np.int64`; `int(...)` casts it to native Python `int`.
- **`st.subheader(...)`**: Renders `<h3>` displaying predicted class and confidence formatted to 2 decimal places (`{conf:.2f}%`).
- **`st.bar_chart(probs)`**: Built-in Vega-Lite/Altair wrapper rendering interactive bars for indices `0..9` (x-axis) vs probability `0..1` (y-axis).

---

### Lines 41–79: Input Mode 2 — Real-World Photo Upload & Alignment Pipeline
```python
else:
    uploaded = st.file_uploader("Upload a digit image (png/jpg)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("L")
        st.image(img, caption="Uploaded image", width=150)
```
- **`st.file_uploader(label, type=["png", "jpg", "jpeg"], accept_multiple_files=False)`**:
  - Returns `None` initially until a file is dropped/selected.
  - Once uploaded, returns an `UploadedFile` (a subclass of `io.BytesIO` living in RAM, with `.name`, `.size`, `.type`).
  - `if uploaded is not None:` is mandatory input validation so the code doesn't crash on initial load.
- **`Image.open(uploaded).convert("L")`**:
  - Decodes JPEG/PNG directly from the `BytesIO` stream.
  - `.convert("L")` converts RGB/RGBA to 8-bit grayscale (`0–255`) using the ITU-R 601-2 luma formula: $L = 0.299 R + 0.587 G + 0.114 B$.

```python
        # invert if background is white/light
        arr = np.array(img, dtype=np.float32)
        if np.mean(arr) > 127:
            arr = 255.0 - arr

        # remove gray paper background and boost contrast
        arr = arr - np.percentile(arr, 60)
        arr = np.clip(arr, 0, None)
        if arr.max() > 0:
            arr = (arr / arr.max()) * 255.0
```
- **Why this 3-step signal cleanup is essential:**
  1. **Polarity Check (`np.mean(arr) > 127`):** Handwritten digits on paper have a bright background (`~180–255`) and dark ink (`~0–60`), so mean pixel value is $>127$. Both `load_digits` and `MNIST` were trained with **white digits on pure black (`0`) backgrounds**. `255.0 - arr` flips dark ink to bright values and bright paper to low values.
  2. **Background Floor Subtraction (`arr - np.percentile(arr, 60)`):** Phone photos of paper aren't pure white—shadows make inverted paper sit around `40–80` instead of `0`. Since the digit stroke covers less than ~30–40% of the image, the **60th percentile** represents the brightest residual background paper noise. Subtracting it + `np.clip(arr, 0, None)` forces the entire paper background to **exact `0.0`**!
  3. **Peak Contrast Stretch (`(arr / arr.max()) * 255.0`):** Scales faint pencil/pen strokes up to full `255.0` white intensity.

```python
        # crop around the digit
        coords = np.argwhere(arr > 30)
        if len(coords) > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            arr = arr[y0:y1, x0:x1]
```
- `np.argwhere(arr > 30)` returns `(N, 2)` coordinates `(row, col)` of all foreground ink pixels above noise threshold `30`.
- `coords.min(axis=0)` finds top-left `(y0, x0)`; `coords.max(axis=0) + 1` finds exclusive bottom-right `(y1, x1)` for Python slicing.
- Cropping removes empty paper margins so the digit fills the target frame instead of shrinking to a 2-pixel speck!

```python
        if img_size == 28:
            digit_20 = Image.fromarray(arr.astype(np.uint8)).resize((20, 20), Image.Resampling.LANCZOS)
            canvas = Image.new("L", (28, 28), 0)
            canvas.paste(digit_20, (4, 4))
            final_img = canvas
        else:
            final_img = Image.fromarray(arr.astype(np.uint8)).resize((8, 8), Image.Resampling.LANCZOS)

        x_input = (np.array(final_img, dtype=np.float32).flatten() / 255.0).reshape(1, -1)
```
- **`Image.Resampling.LANCZOS`**: Highest-quality downsampling filter (3-lobe sinc kernel), avoiding jagged nearest-neighbor artifacts when shrinking a 500×500 photo to 20×20 or 8×8.
- **Why `20×20` pasted at `(4, 4)` onto `28×28` black canvas?** That is the **exact LeCun MNIST preprocessing standard**: every MNIST digit fits inside a `20×20` bounding box centered inside a `28×28` black frame with 4 pixels of zero-padding on all sides! Without that 4px border, the digit touches the image edges, which MNIST's MLP weights expect to be zero.

---

## PART 3: Streamlit Architecture & Mastery (L3 Level)

### 3.1 How Streamlit Works Under the Hood
1. **Client-Server WebSocket Architecture:** Running `uv run streamlit run app.py` starts a Tornado/Uvicorn backend HTTP + WebSocket server on port `8501`. The browser loads a React SPA that communicates widget events and receives Protobuf UI diffs over WebSocket.
2. **Top-to-Bottom Execution Model:** Unlike Flask/FastAPI (route handlers) or React (stateful component tree), **Streamlit has no callbacks required for basic UI**. Every time a user moves a slider, clicks a radio button, or uploads a file, Streamlit **re-runs the entire Python script from Line 1 to the last line** in a dedicated thread!
3. **Why this is both great and dangerous:**
   - *Great:* Super simple linear Python code (`x = st.slider(...)`; `use(x)`).
   - *Dangerous:* Expensive operations (like `joblib.load`, `fetch_openml`, or `model.fit`) placed at top-level will re-run on **every single click** unless cached!

### 3.2 Caching Mastery: `@st.cache_data` vs `@st.cache_resource`
This is the #1 L3 production optimization in Streamlit (and a classic interview/viva question):

| Decorator | Intended For | Behavior | Mutation Safety |
|---|---|---|---|
| **`@st.cache_data`** | **Data** (DataFrames, NumPy arrays, API JSON responses, CSV loads) | Hashes function inputs + body; stores **serialized copy** in cache; returns a **fresh copy** on cache hit | Safe to mutate returned DataFrame without corrupting cache |
| **`@st.cache_resource`** | **Unserializable / Heavy Shared Objects** (ML models `joblib.load`, DB connections, PyTorch/TF models) | Stores **single shared object reference** in memory (no copying/pickling overhead) | Thread-safe for read-only inference (`.predict_proba`); never mutate in place! |

**Production Upgrade of `app.py`:**
```python
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_samples():
    with np.load("sample_digits.npz") as f:
        return {k: f[k] for k in f.files}
```
With those two decorators, moving the slider 100 times triggers **zero disk reads**.

### 3.3 `st.session_state` (Persisting State Across Reruns)
Normal Python variables (`count = 0`) reset on every rerun. `st.session_state` is a per-browser-tab dictionary that persists across reruns:
```python
if "history" not in st.session_state:
    st.session_state.history = []
```
- **Widget Key Binding:** Passing `key="my_slider"` automatically syncs that widget's value at `st.session_state.my_slider`.
- **Callbacks (`on_change`, `on_click`):** Execute *before* the script reruns from Line 1—useful for resetting inputs or logging events.

### 3.4 Controlling Reruns with `st.form`
If you have 5 sliders/inputs and don't want the model to re-predict on every single keystroke:
```python
with st.form("predict_form"):
    # widgets inside here do NOT trigger reruns until Submit is clicked
    idx = st.slider("Pick sample", 0, 99, 0)
    submitted = st.form_submit_button("Predict")
if submitted:
    # run inference
    pass
```

### 3.5 Layout, Containers & Media Primitives
- **Columns:** `col1, col2 = st.columns([1, 2])` (side-by-side layout—great for placing Uploaded Image in `col1` and Bar Chart in `col2`!).
- **Sidebar:** `st.sidebar.selectbox(...)` moves controls cleanly into a collapsible left panel.
- **Tabs & Expanders:** `tab1, tab2 = st.tabs(["Sample", "Upload"])`; `with st.expander("See preprocessed 28x28 image"): ...`.
- **Dynamic Placeholders:** `placeholder = st.empty()` lets you overwrite content in-place inside loops (e.g., live training epoch progress).
- **Matplotlib Leak Warning:** Always pass an explicit `fig` (`st.pyplot(fig)`) and call `plt.close(fig)` afterward so long-running Streamlit servers don't leak Matplotlib figures in RAM!
- **Data Display:** `st.dataframe(df)` (interactive sortable/scrollable React grid) vs `st.table(df)` (static HTML table) vs `st.metric("Accuracy", "97.97%", "+0.06%")`.

### 3.6 Deployment & Production Checklist
1. **Local Run:** `uv run streamlit run app.py` (flags: `--server.port 8501 --server.headless true`).
2. **Streamlit Community Cloud (Live Link):** Push repo (`app.py`, `requirements.txt`, `digits_model.pkl`, `mnist_model.pkl`, `sample_digits.npz`) to GitHub → connect repo on `share.streamlit.io` → select `app.py` → get a public `https://*.streamlit.app` URL!
3. **Secrets Management:** Never hardcode API keys; put them in `.streamlit/secrets.toml` (gitignored) and read via `st.secrets["API_KEY"]`.
4. **Watchdog:** Installing `watchdog` replaces polling with OS-level `FSEvents` on macOS so edits to `app.py` trigger instant hot-reloads.
