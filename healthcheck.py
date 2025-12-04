import httpx
import sys

def check():
    try:
        r = httpx.get("http://localhost:8000/health")
        if r.status_code == 200:
            sys.exit(0)
        sys.exit(1)
    except:
        sys.exit(1)

if __name__ == "__main__":
    check()