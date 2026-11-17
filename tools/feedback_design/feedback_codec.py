"""Proposed compact feedback payload for a fixed window of 16 LoRaWAN frames.

This is a standalone wire-format proposal and analytic benchmark component. It
is not integrated into the network learner and is not a standardized LoRaWAN
MAC command. LoRaWAN framing, encryption, authentication and replay protection
are outside this application payload.

Wire layout (bits within the control byte are numbered from the LSB):

    control[5:0]: SNR state-bin index, 0..54, or 55 for unknown
    control[7:6]: 00 -> all received, no following bytes
                  01 -> one loss, one following byte holding index 0..15
                  10 -> two-byte receipt bitmap, little endian
                  11 -> reserved, invalid

Receipt index 0 is the oldest frame; bitmap bit 0 represents that frame. A bit
value of one means received. The encoder chooses the shortest representation.
For one loss the upper four bits of the position byte must be zero. General
bitmaps are valid even when a shorter representation could encode them.

The SNR bin maps directly to pilot.py state_key's SNR coordinate: its known
states are 0..54 and its unknown state is 55. Integration must consume this bin
directly. Reconstructing a synthetic dB value and digitizing it again can change
state boundaries. The bin is not an exact historical SNR measurement. This
format can support the binary-delivery/energy reward online learner, or frozen
policies needing latest SNR state; it cannot preserve the original online
SNR-dependent reward for every historical successful packet.

Protocol preconditions, NOT enforced by this codec:

* Both endpoints know the fixed window length, its ordering and report cadence.
* A response is generated for the uplink opening the current Class A RX1/RX2
  windows, so that fresh external receive-window context supplies its anchor.
  Stale queued or asynchronous reports require explicit identification instead.
* Sequence/session state, LoRaWAN authentication and replay handling must be
  maintained externally. For a current successfully received uplink, the final
  receipt bit must be one. At startup, indices before the first real frame must
  be handled consistently using the external frame counter and ignored.
* This framing assumes distinct logical uplinks, with NbTrans=1 if those labels
  are interpreted as individual physical attempts. Retries sharing a frame
  counter cannot be identified as separate attempts by this bitmap alone.
* Silence or an invalid/missing report is unknown feedback, not success or
  failure. Apply newly resolved labels once, in order, retaining actual stored
  decision and next-decision states. Overlapping reports must be deduplicated.

The codec intentionally accepts every possible 16-bit receipt pattern,
including an all-zero pattern. Protocol-specific anchor checks belong to the
caller and must not be mistaken for guarantees supplied by byte validation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass


WINDOW_SIZE = 16
MAX_SNR_BIN = 55
MODE_ALL_RECEIVED = 0
MODE_ONE_LOSS = 1
MODE_BITMAP = 2
MODE_RESERVED = 3


@dataclass(frozen=True)
class DecodedFeedback:
    """Payload information only; this object supplies no sequence anchor."""

    receipts: tuple[bool, ...]
    snr_bin: int
    mode: int


def encode_feedback(receipts: Sequence[bool], snr_bin: int) -> bytes:
    """Encode exactly 16 Boolean receipts in the shortest canonical format.

    ``receipts`` is oldest first. Integers 0 and 1 are rejected so accidental
    counters cannot silently be encoded as packet outcomes.
    """
    if isinstance(snr_bin, bool) or not isinstance(snr_bin, int):
        raise TypeError("snr_bin must be an integer")
    if not 0 <= snr_bin <= MAX_SNR_BIN:
        raise ValueError("snr_bin must be in 0..55")
    if not isinstance(receipts, Sequence):
        raise TypeError("receipts must be a sequence of exactly 16 booleans")
    if len(receipts) != WINDOW_SIZE:
        raise ValueError("receipts must contain exactly 16 booleans")
    if any(not isinstance(value, bool) for value in receipts):
        raise TypeError("each receipt must be a bool")

    losses = [index for index, received in enumerate(receipts) if not received]
    if not losses:
        return bytes([snr_bin])
    if len(losses) == 1:
        return bytes([(MODE_ONE_LOSS << 6) | snr_bin, losses[0]])
    bitmap = sum(int(received) << index
                 for index, received in enumerate(receipts))
    return bytes([(MODE_BITMAP << 6) | snr_bin]) + bitmap.to_bytes(2, "little")


def decode_feedback(payload: bytes | bytearray | memoryview) -> DecodedFeedback:
    """Validate payload bytes and decode them; external context is still needed.

    Noncanonical full bitmaps are accepted, but reserved modes, invalid SNR
    codes, unexpected lengths and nonzero one-loss padding bits are rejected.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    data = bytes(payload)
    if not data:
        raise ValueError("payload must contain a control byte")
    snr_bin = data[0] & 0x3F
    mode = data[0] >> 6
    if snr_bin > MAX_SNR_BIN:
        raise ValueError("reserved SNR bin")
    if mode == MODE_RESERVED:
        raise ValueError("reserved feedback mode")
    expected_length = (1, 2, 3)[mode]
    if len(data) != expected_length:
        raise ValueError(f"mode {mode} requires {expected_length} payload bytes")

    if mode == MODE_ALL_RECEIVED:
        receipts = (True,) * WINDOW_SIZE
    elif mode == MODE_ONE_LOSS:
        if data[1] >= WINDOW_SIZE:
            raise ValueError("one-loss position must be 0..15; padding must be zero")
        receipts = tuple(index != data[1] for index in range(WINDOW_SIZE))
    else:
        bitmap = int.from_bytes(data[1:3], "little")
        receipts = tuple(bool(bitmap & (1 << index))
                         for index in range(WINDOW_SIZE))
    return DecodedFeedback(receipts=receipts, snr_bin=snr_bin, mode=mode)


def self_check() -> dict[str, int]:
    """Exercise all 65,536 bitmaps, every SNR code, and malformed payloads."""
    lengths = {1: 0, 2: 0, 3: 0}
    for bitmap in range(1 << WINDOW_SIZE):
        receipts = tuple(bool(bitmap & (1 << index))
                         for index in range(WINDOW_SIZE))
        snr_bin = bitmap % (MAX_SNR_BIN + 1)
        payload = encode_feedback(receipts, snr_bin)
        decoded = decode_feedback(payload)
        if decoded.receipts != receipts or decoded.snr_bin != snr_bin:
            raise AssertionError(f"round trip failed for bitmap {bitmap}")
        loss_count = WINDOW_SIZE - bitmap.bit_count()
        expected_length = 1 if loss_count == 0 else 2 if loss_count == 1 else 3
        if len(payload) != expected_length or decoded.mode != expected_length - 1:
            raise AssertionError(f"noncanonical encoding for bitmap {bitmap}")
        if encode_feedback(decoded.receipts, decoded.snr_bin) != payload:
            raise AssertionError(f"re-encoding changed bitmap {bitmap}")
        lengths[len(payload)] += 1

    # All state-bin codes must also work with each encoder mode.
    for snr_bin in range(MAX_SNR_BIN + 1):
        for bitmap in (0xFFFF, 0xFFFE, 0x0000):
            receipts = tuple(bool(bitmap & (1 << index))
                             for index in range(WINDOW_SIZE))
            if decode_feedback(encode_feedback(receipts, snr_bin)).snr_bin != snr_bin:
                raise AssertionError("SNR code did not survive encoding")

    # These assert byte order and receipt order independently of round trips.
    if decode_feedback(bytes([0x80, 0x01, 0x80])).receipts != (
            True, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, True):
        raise AssertionError("bitmap byte or bit order is incorrect")
    if decode_feedback(bytes([0x40, 15])).receipts[-1] is not False:
        raise AssertionError("one-loss index order is incorrect")

    invalid_payloads = [b"", b"\x00\x00", b"\x40", b"\x40\x10",
                        b"\x40\xff", b"\x40\x00\x00", b"\x80",
                        b"\x80\x00", b"\x80\x00\x00\x00",
                        b"\xc0", b"\xc0\x00", b"\xc0\x00\x00"]
    for snr_bin in range(MAX_SNR_BIN + 1, 64):
        invalid_payloads.extend((bytes([snr_bin]),
                                 bytes([0x40 | snr_bin, 0]),
                                 bytes([0x80 | snr_bin, 0, 0])))
    for payload in invalid_payloads:
        try:
            decode_feedback(payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed payload accepted: {payload.hex()}")

    invalid_encodes = [((True,) * 15, 0), ((True,) * 17, 0),
                       ((True,) * 16, -1), ((True,) * 16, 56),
                       ((True,) * 16, True), ((True,) * 16, 1.0),
                       ((0,) * 16, 0), ((None,) * 16, 0)]
    for receipts, snr_bin in invalid_encodes:
        try:
            encode_feedback(receipts, snr_bin)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid encoder input was accepted")
    for invalid_type in (None, 3, "\x00", [0]):
        try:
            decode_feedback(invalid_type)
        except TypeError:
            pass
        else:
            raise AssertionError("non-bytes decoder input was accepted")

    return {"bitmap_patterns": 1 << WINDOW_SIZE,
            "snr_codes": MAX_SNR_BIN + 1,
            "one_byte_patterns": lengths[1],
            "two_byte_patterns": lengths[2],
            "three_byte_patterns": lengths[3],
            "malformed_payloads_rejected": len(invalid_payloads),
            "invalid_encoder_inputs_rejected": len(invalid_encodes),
            "invalid_decoder_types_rejected": 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true",
                        help="verify all 65,536 receipt patterns and invalid cases")
    args = parser.parse_args()
    if args.self_check:
        import json
        print(json.dumps(self_check(), indent=2))
    else:
        parser.print_help()
