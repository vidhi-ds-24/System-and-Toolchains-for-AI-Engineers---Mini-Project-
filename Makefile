.DEFAULT_GOAL := setup

UV ?= uv
IDS := wenyinh cosettez vidhikot mbbhatt
TEAM := power rangers STC
DATA_URL := https://kitchin-services.cheme.cmu.edu/f26-06763/data
DATA_FILES := tep_fault_free_training.parquet tep_faulty_training_runs01-20.parquet
EVIDENCE_URL := https://kitchingroup.cheme.cmu.edu/f26-06763/miniproject-evidence.py

.PHONY: setup data ridge ridge-score ridge-plot ridge-evaluate ridge-diagnose evidence pca pca-evaluate pca-diagnose pca-plot

setup:
	$(UV) venv --allow-existing --prompt miniproj .venv
	$(UV) lock
	$(UV) sync --locked

data:
	@mkdir -p data
	@set -eu; for file in SHA256SUMS $(DATA_FILES); do \
		if [ ! -f "data/$$file" ]; then \
			curl -fL "$(DATA_URL)/$$file" -o "data/$$file.part"; \
			mv "data/$$file.part" "data/$$file"; \
		fi; \
	done
	@cd data && shasum -a 256 -c SHA256SUMS

ridge:
	$(UV) run python -m src.ridge

ridge-score:
	$(UV) run python -m src.run_forecast_score

ridge-plot:
	$(UV) run python -m src.plot_ridge_score

ridge-evaluate:
	$(UV) run python -m src.evaluate_ridge

ridge-diagnose: ridge-evaluate
	$(UV) run python -m src.diagnose_ridge

evidence:
	$(UV) run --no-project $(EVIDENCE_URL) --andrew-ids $(IDS) --team "$(TEAM)"

pca: data
	$(UV) run python src/pca.py

pca-evaluate:
	$(UV) run python -m src.evaluate_pca

pca-diagnose: pca-evaluate
	$(UV) run python -m src.diagnose_pca

pca-plot: pca-evaluate
	$(UV) run python -m src.plot_pca
