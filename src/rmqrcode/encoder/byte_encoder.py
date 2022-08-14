from .encoder_base import EncoderBase


class ByteEncoder(EncoderBase):
    @classmethod
    def mode_indicator(cls):
        return "011"

    @classmethod
    def encode(cls, data, character_count_indicator_length):
        res = cls.mode_indicator()
        res += bin(len(data))[2:].zfill(character_count_indicator_length)
        res += cls._encoded_bits(data)
        return res

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
