
# YOLO3D Project Makefile

.PHONY: help setup install train train-fast eval serve test-api clean clean-logs lint format test test-full sync

# Default target
help:
	@echo "YOLO3D Project Commands:"
	@echo "  setup        Setup conda environment"
	@echo "  install      Install dependencies with pip"  
	@echo "  train        Train YOLO3D model"
	@echo "  train-fast   Fast training (10 epochs)"
	@echo "  eval         Evaluate model"
	@echo "  serve        Start FastAPI server"
	@echo "  test-api     Test API endpoints"
	@echo "  test         Run pytest tests"
	@echo "  test-full    Run all tests"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with pre-commit"
	@echo "  clean        Clean cache and logs"
	@echo "  clean-logs   Clean logs only"
	@echo "  sync         Sync with main branch"

setup:
	@echo "🚀 Setting up YOLO3D environment..."
	./setup.sh

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.devel.txt

train:
	@echo "🏋️ Training YOLO3D model..."
	python src/train.py --config-name=train_yolo3d

train-fast:
	@echo "⚡ Fast training (10 epochs)..."
	python src/train.py --config-name=train_yolo3d trainer.max_epochs=10

eval:
	@echo "📊 Evaluating model..."
	python src/eval.py --config-name=eval

serve:
	@echo "🚀 Starting FastAPI server..."
	PYTHONPATH=./ python src/serve.py

test-api:
	@echo "🧪 Testing API..."
	python tests/test_api.py --create-dummy

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v -k "not slow"

test-full:
	@echo "🧪 Running all tests..."
	pytest tests/ -v

lint:
	@echo "🔍 Linting code..."
	@/bin/python -m black --check --line-length 99 src/ serve.py test_api.py || echo "Some files need formatting (run 'make format' to fix)"
	@/bin/python -m isort --check-only --profile black src/ serve.py test_api.py || echo "Import sorting needed (run 'make format' to fix)"
	@/bin/python -m flake8 --extend-ignore=E203,E402,E501,F401,F841 --exclude=logs/*,data/* src/ serve.py test_api.py || echo "Code style issues found"

format:
	@echo "🎨 Formatting code..."
	@/bin/python -m black --line-length 99 src/ serve.py test_api.py
	@/bin/python -m isort --profile black src/ serve.py test_api.py
	@echo "✅ Code formatting complete!"

clean:
	@echo "🧹 Cleaning cache and logs..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.DS_Store" -delete
	rm -rf .pytest_cache/
	rm -rf logs/train/
	rm -rf dummy_test_image.jpg
	rm -rf dist/
	rm -f .coverage

clean-logs:
	@echo "🧹 Cleaning logs only..."
	rm -rf logs/**

sync:
	@echo "🔄 Syncing with main branch..."
	git pull
	git pull origin main
