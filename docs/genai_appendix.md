# GenAI Usage Appendix

All use of generative AI tools in this project is documented below per assignment requirements.

---

Prompt: Generate a Kafka producer that connects to a WebSocket and publishes to a topic
Used in: scripts/ws_ingest.py
Verification: Replaced placeholder connection logic with the actual Coinbase Advanced Trade
WebSocket URL. Added reconnect and heartbeat handling manually after testing showed the
generated stub did not handle disconnects.

---

Prompt: Write docstrings for all functions in featurizer.py
Used in: features/featurizer.py
Verification: Read each docstring and corrected inaccurate parameter descriptions.
Removed one docstring that referred to a function that no longer existed after refactoring.

---

Prompt: Explain the difference between PR-AUC and ROC-AUC for imbalanced classification
Used in: docs/model_card_v1.md and docs/scoping_brief.md
Verification: Cross-checked against scikit-learn documentation and course lecture notes.
Used to justify metric selection in the model card.

---

Prompt: Generate an XGBoost training loop with MLflow logging and time-based train/val/test splits
Used in: models/train.py
Verification: Updated all feature column names to match the actual data schema. Added the
volatility threshold baseline class manually. Fixed MLflow artifact path errors that appeared
during testing.

---

Prompt: Write an infer.py script that loads a saved XGBoost model and benchmarks inference speed
Used in: models/infer.py
Verification: Tested against the actual test parquet file. Confirmed benchmark output
satisfies the assignment requirement of less than 2x real-time for a 60-second window.

---

Prompt: Generate an Evidently drift report script comparing two time windows of a parquet file
Used in: scripts/evidently_model_report.py
Verification: Fixed import errors caused by breaking API changes in Evidently 0.7.x.
Confirmed the HTML report renders with correct per-feature drift statistics.

---

Prompt: Draft a model card for a binary classifier trained on financial time series data
Used in: docs/model_card_v1.md
Verification: Replaced all placeholder values with actual project metrics, feature names,
threshold values, and drift results from the Evidently report.

---

Summary: GenAI tools were used for boilerplate scaffolding, docstring generation, and
drafting structured documents. In all cases the output was reviewed, tested against real
data, and edited to reflect actual project decisions. No generated code was used without
being read and verified by the author.