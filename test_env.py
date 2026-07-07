from __future__ import annotations

from importlib import import_module


def _load_optional(module_name: str):
	try:
		return import_module(module_name), None
	except Exception as exc:
		return None, exc


mediapipe, mediapipe_error = _load_optional("mediapipe")
torch, torch_error = _load_optional("torch")
torchvision, torchvision_error = _load_optional("torchvision")

if mediapipe is None:
	print(f"! mediapipe: unavailable ({mediapipe_error})")
else:
	print("✓ mediapipe:", mediapipe.__version__)

if torch is None:
	print("! torch: not installed")
else:
	print("✓ torch:", torch.__version__)
	print("✓ GPU available:", torch.cuda.is_available())

if torchvision is None:
	print("! torchvision: not installed")
else:
	print("✓ torchvision:", torchvision.__version__)

print()
print("Training environment check complete.")
