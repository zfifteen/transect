.PHONY: bench bench-drift analyze clean help

help:
	@echo "TRANSEC Benchmark Targets:"
	@echo "  make bench        - Run baseline benchmarks (no drift)"
	@echo "  make bench-drift  - Run benchmarks with ±3 slot clock skew"
	@echo "  make analyze      - Parse logs and generate CSV + plots"
	@echo "  make clean        - Remove benchmark outputs"

bench:
	python examples/transec_udp_demo.py benchmark --count 200 --prime_strategy none --drift_window 3 --slot_duration 0.050 --skew_slots 0 --out baseline.log
	python examples/transec_udp_demo.py benchmark --count 200 --prime_strategy nearest --drift_window 3 --slot_duration 0.050 --skew_slots 0 --out prime.log

bench-drift:
	python examples/transec_udp_demo.py benchmark --count 200 --prime_strategy none --drift_window 3 --slot_duration 0.050 --skew_slots 3 --out baseline_drift.log
	python examples/transec_udp_demo.py benchmark --count 200 --prime_strategy nearest --drift_window 3 --slot_duration 0.050 --skew_slots 3 --out prime_drift.log

analyze:
	python scripts/parse_transect_logs.py baseline*.log prime*.log --out results.csv --plot plots/drift-hist.png

clean:
	rm -f *.log results.csv
	rm -rf plots/
