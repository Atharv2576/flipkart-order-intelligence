# faiss and torch each bundle their own OpenMP runtime; on macOS, importing
# faiss before torch has ever been imported causes a segfault the first time
# a real torch computation runs. Importing torch first here, before any
# part3 submodule can import faiss, makes torch's runtime win consistently
# regardless of which submodule a caller imports first.
import torch  # noqa: F401
