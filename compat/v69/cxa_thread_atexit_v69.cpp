// Compatibility shim for linking newer Hexagon Tools libc++abi against older
// vendor DSP C runtimes that do not export __cxa_thread_atexit_impl.
// Scope this object to standalone launchers or packages that actually require
// the symbol; it is not part of the MLIR kernel ABI.

#include <cstdlib>

namespace {
struct Entry {
  void (*destructor)(void*);
  void* object;
};
Entry entries[128];
unsigned count = 0;
void run_destructors() {
  while (count) {
    Entry entry = entries[--count];
    entry.destructor(entry.object);
  }
}
}  // namespace

extern "C" int __cxa_thread_atexit_impl(void (*destructor)(void*), void* object,
                                          void*) {
  if (count >= 128) return -1;
  if (count == 0) std::atexit(run_destructors);
  entries[count++] = {destructor, object};
  return 0;
}
