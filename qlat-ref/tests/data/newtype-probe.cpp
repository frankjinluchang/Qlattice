// Emit RngState::newtype() test vectors as JSON (newtype-vectors.json).
//
// newtype() is C++-only (qlat's Python binding does not expose it), so these
// vectors cannot be produced from qlat's Python API.  They pin the
// "[type,index]" counter string and the tag-clearing behavior of split(), which
// qlat_ref.rng exposes as a documented extension.  Regenerate with:
//
//   g++ -O2 -std=c++17 -I <repo>/qlat-utils/qlat_utils/include \
//       newtype-probe.cpp \
//       <repo>/qlat-utils/qlat_utils/lib/sha256.cpp \
//       <repo>/qlat-utils/qlat_utils/lib/rng-state.cpp -o newtype-probe
//   ./newtype-probe > newtype-vectors.json
//
// The Timer::display_stack_always() stub below stands in for the full
// qlat_utils library, which qassert() references; only the RNG translation
// units are needed.
#include <qlat-utils/rng-state.h>
#include <qlat-utils/timer.h>

#include <cstdio>

namespace qlat
{  //
// Stub: qassert() in rng-state.cpp references the timer's stack dump.
void Timer::display_stack_always() {}
}  // namespace qlat

static void print_u64_array(const qlat::RngState& rs, int n)
{
  using namespace qlat;
  RngState r = rs;
  printf("[");
  for (int i = 0; i < n; ++i) {
    printf("%s%lu", i == 0 ? "" : ", ", (unsigned long)rand_gen(r));
  }
  printf("]");
}

int main()
{
  using namespace qlat;
  const int n = 8;
  printf("{\n");
  printf("  \"newtype_cases\": {\n");
  // type tag 5 on a seeded state
  RngState base = RngState("test-rng");
  RngState typed = base.newtype(5);
  printf("    \"typed_5\": {\n");
  printf(
      "      \"note\": \"RngState('test-rng').newtype(5): the counter string "
      "becomes "
      "'[type,index]'\",\n");
  printf(
      "      \"seed_spec\": {\"kind\": \"seed\", \"value\": \"test-rng\"},\n");
  printf("      \"type\": 5,\n");
  printf("      \"values\": ");
  print_u64_array(typed, n);
  printf("\n    },\n");
  // newtype on the fresh (unseeded) state, then split it
  RngState fresh = RngState();
  RngState typed_fresh = fresh.newtype(0);
  printf("    \"typed_0_split_k\": {\n");
  printf(
      "      \"note\": \"RngState().newtype(0).split('k'): the split label "
      "becomes "
      "'[type,index] {k}' and the tag is cleared\",\n");
  printf("      \"seed_spec\": {\"kind\": \"fresh\"},\n");
  printf("      \"type\": 0,\n");
  printf("      \"split\": \"k\",\n");
  printf("      \"num_bytes_after_split\": %lu,\n",
         (unsigned long)typed_fresh.split("k").numBytes);
  printf("      \"values\": ");
  print_u64_array(typed_fresh.split("k"), n);
  printf("\n    }\n");
  printf("  }\n");
  printf("}\n");
  return 0;
}
