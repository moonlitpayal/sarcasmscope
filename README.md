# SARcasmScope-Civic test set (identifiers and labels)

206 public YouTube comments on civic issues in Pune and Hyderabad (metro, traffic, potholes, flooding, water supply, garbage) in English, Hindi/Hinglish, Marathi and Telugu, each labelled for sarcasm by three annotators.

**Comment text is not included**, in line with YouTube's terms of service. To reproduce the study, re-fetch the comments that are still public with the YouTube Data API v3 (`comments.list` with `id=<comment_id>`).

## Files
- `civic_test_ids_labels.csv` — one row per comment
- `annotation_guideline.md` — the guideline the annotators followed

## Columns
| Column | Meaning |
|---|---|
| item_id | Item number in the study (SQ-xxxx) |
| comment_id | YouTube comment ID |
| video_id | YouTube video ID |
| language | Language of the comment (assigned automatically from script and function words) |
| script | Latin (Roman), Devanagari, Telugu or Mixed |
| topic | Civic topic of the video |
| sampling | `random` or `cue-seeded` (chosen because it contained a common sarcasm cue) |
| is_reply | 1 if the comment replies to another comment |
| annotator_1–3 | Each annotator's label: Yes (sarcastic), No, or blank |
| label | Majority label: 1 = sarcastic, 0 = not sarcastic |
| agreement | `unanimous` (3–0) or `majority` (2–1) |

No commenter names or channel identifiers were stored.
