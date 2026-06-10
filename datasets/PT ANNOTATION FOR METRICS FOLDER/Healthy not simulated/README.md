# PT Annotation CSV Package

Created from the provided PT annotation forms and user-confirmed assumptions.

Assumptions encoded:
- Participants and affected sides:
  - PT_01: Right
  - PT_02: Left
  - PT_03: Right
  - PT_04: Left
  - PT_05: Left
- Exercises for each participant:
  - Abduction
  - Flexion
- Sets per exercise: 2
- Repetitions per set: 10
- Correctness / ROM label: all `correct`
- Trunk Lean: all `No`
- Shoulder Hiking: all `No`
- Tracking artifact / BI exclusion: all `No`
- Micro-break needed: all `No`

Files:
- `pt_annotations_all_standard.csv`: complete combined annotation file with full fields.
- `pt_annotations_all_compute_metrics.csv`: compatibility combined file with `PT_hiking`.
- `participant_metadata.csv`: participant-side mapping and inclusion notes.
- `pt_microbreak_set_annotations_all.csv`: set-level micro-break labels.
- Per-participant folders contain separate Abduction and Flexion annotation CSV files.

Important note:
These rows are based on the stated annotation summary. If any paper form has a marked BI/boundary index, unscorable rep, or micro-break recommendation, update the corresponding row before running final metrics.
