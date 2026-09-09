"""Fresh-process public-API arm; the reference arm runs iobrpy.main directly."""
import json
from pathlib import Path
import sys
import time

import iobrx


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    start = time.perf_counter()
    result = getattr(iobrx, request["function"])(**request["parameters"])
    elapsed = time.perf_counter() - start
    rc = int(result.get("rc", 0)) if isinstance(result, dict) else 0
    print(json.dumps({"api_seconds": elapsed, "return_code": rc,
                      "iobrx": iobrx.__version__, "backend": iobrx.backend_info()}))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
