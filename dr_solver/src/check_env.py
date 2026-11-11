"""Phase 0: print the environment.  python -m src.check_env --data /path/to/data"""
import argparse
import platform
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    a = ap.parse_args()
    print("OS:", platform.platform())
    print("Python:", sys.version.split()[0])
    try:
        import torch
        print("torch:", torch.__version__)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB")
        else:
            print("GPU: none (training will be slow; use --size 256)")
    except ImportError:
        print("torch not installed")
    for mod in ("timm", "ortools", "cv2", "sklearn", "pandas"):
        try:
            m = __import__(mod)
            print(f"{mod}: {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{mod}: MISSING")
    du = shutil.disk_usage(a.data)
    print(f"free disk at {a.data}: {du.free/1e9:.1f} GB")


if __name__ == "__main__":
    main()
