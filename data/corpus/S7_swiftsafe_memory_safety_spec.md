# SwiftSafe System Architecture & Memory Safety Guidelines

## Overview
SwiftSafe is a safe system programming framework designed to eliminate memory corruption vulnerabilities (use-after-free, double free, buffer overflow) in native binaries through compile-time ownership tracking and automated static inspection.

## Memory Safety Guarantees
1. **Zero Raw Pointers in Public API:** All reference passing must use safe wrappers (`SafeRef<T>` or `UniquePtr<T>`).
2. **Bounds Checking:** Direct array indexing compiles to runtime bounds-checked assertions unless proven safe by static loop bounds invariants.
3. **Use-After-Free Prevention:** Lifetime annotations enforce that borrowed references cannot outlive their parent scope owner.

## Verification Engine
SwiftSafe integrates static analysis checks within the CI pipeline:
- Static analysis pass checks for concurrency race conditions on shared global memory.
- Safe dynamic memory allocations require `SwiftSafeAllocator::allocate_aligned(size_t bytes, size_t alignment)`.
