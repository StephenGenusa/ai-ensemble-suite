# Thread Safety of llama-cpp-python

llama-cpp-python is generally **not fully thread-safe** in its core operations. Here's what you should know:

## Thread Safety Considerations

1. **Core model operations**: The main inference operations are not designed to be called concurrently from multiple threads on the same model instance.

2. **Model instances**: Different model instances can be used in separate threads, but a single model instance should not be shared across threads for concurrent inference.

3. **Batch processing**: The library does have some internal threading capabilities for batch processing, but this is controlled within the library itself.

4. **Global state**: Some components of the underlying llama.cpp maintain global state that can cause issues in multi-threaded environments.

## Safe Usage Patterns

- Create separate model instances for separate threads
- Use locks/mutexes if you must share a model instance across threads
- Consider using a queue-based architecture where a dedicated thread handles all model interactions

If you need concurrent processing, it's generally better to use a process-based approach (like multiple workers) rather than attempting to share the model across threads.