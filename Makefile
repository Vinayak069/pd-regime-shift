EXPS ?= control crisis_2008 covid
LOANS ?=            # e.g. LOANS=20000 to sample loans per vintage when building the panel
PY = PYTHONPATH=src python3

fred:
	$(PY) -m pdshift.macro --download
panel:
	$(PY) -m pdshift.panel $(if $(LOANS),--loans-per-vintage $(LOANS))
experiments:
	for e in $(EXPS); do $(PY) -m pdshift.experiment --exp $$e --seeds 5 --boot 1000 || exit 1; done
select:
	for e in $(EXPS); do $(PY) -m pdshift.selection --exp $$e || exit 1; done
robust:
	for e in $(EXPS); do $(PY) -m pdshift.robustness --exp $$e --seeds 5 --boot 1000 || exit 1; done
smoke:   # fast run of one experiment, one seed, few bootstrap draws
	$(PY) -m pdshift.experiment --exp control --seeds 1 --boot 20 --lifetime-months 60
test:
	PYTHONPATH=src python3 -m pytest -q tests
.PHONY: fred panel experiments select robust smoke test
