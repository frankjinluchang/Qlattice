#pragma once

#include <qlat-utils/show.h>

#include <cstdint>
#include <string>
#include <vector>

namespace qlat
{  //

struct API RngState;

void reset(RngState& rs);

void reset(RngState& rs, const std::string& seed);

void reset(RngState& rs, const Long seed);

void split_rng_state(RngState& rs, const RngState& rs0,
                     const std::string& sindex);

void split_rng_state(RngState& rs, const RngState& rs0, const Long sindex = 0);

void set_type(RngState& rs, const uint64_t type);

uint64_t rand_gen(RngState& rs);

RealD u_rand_gen(RngState& rs, const RealD lower = 0.0,
                 const RealD upper = 1.0);

RealD g_rand_gen(RngState& rs, const RealD center = 0.0,
                 const RealD sigma = 1.0);

void compute_hash_with_input(uint32_t hash[8], const RngState& rs,
                             const std::string& input);

// --------------------
// State serialization (see docs/qlat-utils/qlat_rng_state.md).
// The byte layout is part of the cross-implementation contract with qlat_ref:
// 8 byte header "QLATRNG" + version 0x01, then the 24 uint32 words of
// exportRngState() in little-endian order (104 bytes total).
constexpr size_t RNG_STATE_BYTES_HEADER_SIZE = 8;
constexpr size_t RNG_STATE_BYTES_NUM_OF_WORD = 24;
constexpr size_t RNG_STATE_BYTES_SIZE =
    RNG_STATE_BYTES_HEADER_SIZE + 4 * RNG_STATE_BYTES_NUM_OF_WORD;

std::string get_rng_state_bytes(const RngState& rs);

void set_rng_state_bytes(RngState& rs, const std::string& data);

struct API RngState {
  uint64_t numBytes;
  uint32_t hash[8];
  uint64_t type;
  uint64_t index;
  //
  uint64_t cache[3];
  RealD gaussian;
  Int cacheAvail;
  bool gaussianAvail;
  //
  inline void init() { reset(*this); }
  //
  RngState() { init(); }
  RngState(const std::string& seed) { reset(*this, seed); }
  RngState(const Long seed) { reset(*this, seed); }
  RngState(const RngState& rs0, const std::string& sindex)
  {
    memset((void*)this, 0, sizeof(RngState));
    split_rng_state(*this, rs0, sindex);
  }
  RngState(const RngState& rs0, const Long sindex)
  {
    memset((void*)this, 0, sizeof(RngState));
    split_rng_state(*this, rs0, sindex);
  }
  //
  RngState split(const std::string& sindex) const
  {
    return RngState(*this, sindex);
  }
  RngState split(const Long sindex) const { return RngState(*this, sindex); }
  //
  RngState newtype(const uint64_t type) const
  {
    RngState rs(*this);
    set_type(rs, type);
    return rs;
  }
};

API inline RngState& get_global_rng_state()
{
  static RngState rs;
  return rs;
}

template <class M>
void random_permute(std::vector<M>& vec, const RngState& rs_)
{
  RngState rs = rs_;
  const Long size = (Long)vec.size();
  M tmp;
  for (Long k = 0; k < size; ++k) {
    const Long kk = rand_gen(rs) % (size - k);
    tmp = vec[k];
    vec[k] = vec[k + kk];
    vec[k + kk] = tmp;
  }
}

}  // namespace qlat
