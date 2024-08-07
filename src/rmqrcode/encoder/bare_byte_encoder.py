from .encoder_base import EncoderBase


class BareByteEncoder(EncoderBase):
    @classmethod
    def mode_indicator(cls):
        return "011"

    @classmethod
    def _encoded_bits(cls, encoded):
        res = ""
        for byte in encoded:
            res += bin(byte)[2:].zfill(8)
        return res

    @classmethod
    def length(cls, data, character_count_indicator_length):
        return len(cls.mode_indicator()) + character_count_indicator_length + 8 * len(data)

    @classmethod
    def characters_num(cls, data):
        return len(data)

    @classmethod
    def is_valid_characters(cls, data):
        return True  # Any characters can encode in the Byte Mode
