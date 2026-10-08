SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command

PYTHON := .venv\Scripts\python.exe

.PHONY: help install check collect train run clean

help:
	@Write-Host ""
	@Write-Host "Gesture Desktop Control"
	@Write-Host "======================="
	@Write-Host ""
	@Write-Host "GNU Make is optional."
	@Write-Host "Windows users can also run the Python files directly."
	@Write-Host ""
	@Write-Host "Commands:"
	@Write-Host "  make install  - Install Python dependencies"
	@Write-Host "  make check    - Check dependencies and required files"
	@Write-Host "  make collect  - Collect gesture training data"
	@Write-Host "  make train    - Train the gesture classifier"
	@Write-Host "  make run      - Run the desktop controller"
	@Write-Host "  make clean    - Remove generated training files"
	@Write-Host ""

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

check:
	@if (!(Test-Path '$(PYTHON)')) { Write-Error 'Virtual environment not found. Create it with: python -m venv .venv'; exit 1 }
	$(PYTHON) -c "import cv2, mediapipe, numpy, pandas, sklearn, PIL, win32com.client, keyboard; print('Python dependencies OK')"
	@if (Test-Path 'hand_landmarker.task') { Write-Host 'hand_landmarker.task OK' } else { Write-Error 'Missing hand_landmarker.task'; exit 1 }
	@if (Test-Path 'gesture_data.csv') { Write-Host 'gesture_data.csv OK' } else { Write-Host 'gesture_data.csv not found - run make collect first' }
	@if (Test-Path 'gesture_model.pkl') { Write-Host 'gesture_model.pkl OK' } else { Write-Host 'gesture_model.pkl not found - run make train first' }

collect:
	@if (!(Test-Path 'hand_landmarker.task')) { Write-Error 'Missing hand_landmarker.task'; exit 1 }
	$(PYTHON) collectData.py

train:
	@if (!(Test-Path 'gesture_data.csv')) { Write-Error 'Missing gesture_data.csv. Run make collect first.'; exit 1 }
	$(PYTHON) train_gesture.py

run:
	@if (!(Test-Path 'hand_landmarker.task')) { Write-Error 'Missing hand_landmarker.task'; exit 1 }
	@if (!(Test-Path 'gesture_model.pkl')) { Write-Error 'Missing gesture_model.pkl. Run make train first.'; exit 1 }
	$(PYTHON) main.py

clean:
	Remove-Item -Force -ErrorAction SilentlyContinue gesture_data.csv
	Remove-Item -Force -ErrorAction SilentlyContinue gesture_model.pkl
	Write-Host "Generated training files removed."