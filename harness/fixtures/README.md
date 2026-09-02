# fixture

A tiny, frozen directory the eval tasks run against.

Frozen on purpose: ground truth like "there are 3 .py files" stops being true
the moment someone adds a file. The eval harness must not measure a moving
target - same reason rag-eval lives outside the corpus it indexes.
