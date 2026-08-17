import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # workaround for the 0xC0000005 crash --
# must be set before torch/sentence-transformers get imported below, which is why
# this is the very first thing in the file.

import time
from verityrag.embedding import SentenceTransformerEmbedder

emb = SentenceTransformerEmbedder()
print("model loaded, dim =", emb.dim, flush=True)

texts = [f"This is test sentence number {i} with a few extra words to pad it out." for i in range(845)]

batch_size = 50
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    t0 = time.time()
    vecs = emb.embed(batch)
    print(f"batch {i}-{i+len(batch)}: shape={vecs.shape} in {time.time()-t0:.2f}s", flush=True)

print("ALL DONE", flush=True)