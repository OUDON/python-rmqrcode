from .encoder_base import EncoderBase


class ByteEncoder(EncoderBase):
    @classmethod
    def mode_indicator(cls):
        return "011"

    @classmethod
    def _encoded_bits(cls, s):
        res = ""
        encoded = s.encode("utf-8")
        for byte in encoded:
            res += bin(byte)[2:].zfill(8)
        return res

    @classmethod
    def length(cls, data, character_count_indicator_length):
        return len(cls.mode_indicator()) + character_count_indicator_length + 8 * len(data.encode("utf-8"))

    @classmethod
    def characters_num(cls, data):
        return len(data.encode("utf-8"))

    @classmethod
    def is_valid_characters(cls, data):
        return not any(ord(c) & 0x80 for c in data)
