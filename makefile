.PHONY: run-api run-tests run-all

export-tc-model:
	cd exporters && uv run export-tc-model.py

export-tl-model:
	cd exporters && uv run export_translation_model.py

run-api:
	cd api && cargo run

run-tests:
	cd tests/model_tests && cargo run --bin test-yelp
	cd tests/model_tests && uv run test_yelp_python.py

run-all:
	make run-api
	make run-tests
