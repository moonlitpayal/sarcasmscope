# Fine-tuning MuRIL / XLM-R / mBERT on Google Colab (free GPU)

Your laptop has the CPU-only build of PyTorch, so transformer training should run on Colab.
The TF-IDF baseline, the audit, the splits and the web app all run fine on your laptop.

1. Colab -> Runtime -> Change runtime type -> **T4 GPU**.
2. Upload `sarcasmscope.zip` (and your dataset file) to Colab, or mount Google Drive.
3. Run these cells in order:

```python
!unzip -q sarcasmscope.zip -d /content && cd /content/sarcasmscope && pip install -q -e . torch transformers
```
```python
%cd /content/sarcasmscope
!python scripts/smoke_transformer.py xlmr          # must end with: SMOKE TEST PASSED
```
```python
!sarcasmscope run --config configs/my_multilingual.yaml     # development run: VALIDATION split only
```
4. Download `runs/<run_name>/` (report.md, tables/, bundles/) to your laptop.
5. **Only at the very end**, when the design is frozen: `!sarcasmscope run --config ... --final` (once).

Rough time on a T4 (indicative, depends on data size): a 5k-sentence dataset takes a few minutes per model per seed.
Start with one seed, then raise `seeds` to 3-5 for reported numbers.

To use a fine-tuned model in the web app on your laptop:
`sarcasmscope serve --bundle runs/<run_name>/bundles/muril_seed42` (the bundle is a folder; it needs `pip install torch transformers`).
