.PHONY: bench bench-drift analyze clean help server-none server-nearest

help:
	@echo "TRANSEC Benchmark Targets:"
	@echo "  make server-none      - Start server with prime_strategy=none"
	@echo "  make server-nearest   - Start server with prime_strategy=nearest" 
	@echo "  make bench            - Run baseline benchmarks (no drift)"
	@echo "  make bench-drift      - Run benchmarks with ±3 slot clock skew"
	@echo "  make analyze          - Parse logs and generate CSV + plots"
	@echo "  make clean            - Remove benchmark outputs"
	@echo ""
	@echo "Note: Server must be running before benchmarks. For drift tests:"
	@echo "  1. Run 'make server-none' in one terminal"
	@echo "  2. Run 'make bench-drift' to test baseline"
	@echo "  3. Stop server, run 'make server-nearest'"
	@echo "  4. Run benchmarks again with prime strategy"

server-none:
	python examples/transec_udp_demo.py server --slot_duration 0.050 --drift_window 3 --prime_strategy none

server-nearest:
	python examples/transec_udp_demo.py server --slot_duration 0.050 --drift_window 3 --prime_strategy nearest

bench:
	python examples/transec_udp_demo.py benchmark --count 200 --prime_strategy none --drift_window 3 --slot_duration 0.050 --skew_slots 0 --out baseline.log
	python examples/transec_udp_demo.py benchmark --count 200 --prime_strategy nearest --drift_window 3 --slot_duration 0.050 --skew_slots 0 --out prime.log

bench-drift:
	bash scripts/run_benchmark.sh

analyze:
	python scripts/parse_transect_logs.py baseline*.log prime*.log --out results.csv --plot plots/drift-hist.png

clean:
	rm -f *.log results.csv
	rm -rf plots/
