.DEFAULT_GOAL := setup

UV ?= uv
DATA_URL := https://kitchin-services.cheme.cmu.edu/f26-06763/data
DATA_FILES := tep_fault_free_training.parquet tep_faulty_training_runs01-20.parquet
EVIDENCE_URL := https://kitchingroup.cheme.cmu.edu/f26-06763/miniproject-evidence.py

.PHONY: setup data evidence

setup:
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

evidence:
	@test -n "$(TEAM)" || { echo 'Set TEAM="Team name"'; exit 1; }
	@set -- $(IDS); test "$$#" -eq 4 || { echo 'Set IDS="id1 id2 id3 id4"'; exit 1; }
	$(UV) run --no-project $(EVIDENCE_URL) --andrew-ids $(IDS) --team "$(TEAM)"
