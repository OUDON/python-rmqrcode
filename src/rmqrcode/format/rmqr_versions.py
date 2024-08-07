from ..encoder import AlphanumericEncoder, ByteEncoder, BareByteEncoder, KanjiEncoder, NumericEncoder
from .error_correction_level import ErrorCorrectionLevel

rMQRVersions = {
    "R7x43": {
        "version_indicator": 0b00000,
        "height": 7,
        "width": 43,
        "remainder_bits": 0,
        "character_count_indicator_length": {
            NumericEncoder: 4,
            AlphanumericEncoder: 3,
            ByteEncoder: 3,
            BareByteEncoder: 3,
            KanjiEncoder: 2,
        },
        "codewords_total": 13,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 13,
                    "k": 6,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 13,
                    "k": 3,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 48,
            ErrorCorrectionLevel.H: 24,
        },
    },
    "R7x59": {
        "version_indicator": 0b00001,
        "height": 7,
        "width": 59,
        "remainder_bits": 3,
        "character_count_indicator_length": {
            NumericEncoder: 5,
            AlphanumericEncoder: 5,
            ByteEncoder: 4,
            BareByteEncoder: 4,
            KanjiEncoder: 3,
        },
        "codewords_total": 21,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 21,
                    "k": 12,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 21,
                    "k": 7,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 96,
            ErrorCorrectionLevel.H: 56,
        },
    },
    "R7x77": {
        "version_indicator": 0b00010,
        "height": 7,
        "width": 77,
        "remainder_bits": 5,
        "character_count_indicator_length": {
            NumericEncoder: 6,
            AlphanumericEncoder: 5,
            ByteEncoder: 5,
            BareByteEncoder: 5,
            KanjiEncoder: 4,
        },
        "codewords_total": 32,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 32,
                    "k": 20,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 32,
                    "k": 10,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 160,
            ErrorCorrectionLevel.H: 80,
        },
    },
    "R7x99": {
        "version_indicator": 0b00011,
        "height": 7,
        "width": 99,
        "remainder_bits": 6,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 6,
            ByteEncoder: 5,
            BareByteEncoder: 5,
            KanjiEncoder: 5,
        },
        "codewords_total": 44,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 44,
                    "k": 28,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 44,
                    "k": 14,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 224,
            ErrorCorrectionLevel.H: 112,
        },
    },
    "R7x139": {
        "version_indicator": 0b00100,
        "height": 7,
        "width": 139,
        "remainder_bits": 1,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 6,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 5,
        },
        "codewords_total": 68,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 68,
                    "k": 44,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 2,
                    "c": 34,
                    "k": 12,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 352,
            ErrorCorrectionLevel.H: 192,
        },
    },
    "R9x43": {
        "version_indicator": 0b00101,
        "height": 9,
        "width": 43,
        "remainder_bits": 2,
        "character_count_indicator_length": {
            NumericEncoder: 5,
            AlphanumericEncoder: 5,
            ByteEncoder: 4,
            BareByteEncoder: 4,
            KanjiEncoder: 3,
        },
        "codewords_total": 21,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 21,
                    "k": 12,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 21,
                    "k": 7,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 96,
            ErrorCorrectionLevel.H: 56,
        },
    },
    "R9x59": {
        "version_indicator": 0b00110,
        "height": 9,
        "width": 59,
        "remainder_bits": 3,
        "character_count_indicator_length": {
            NumericEncoder: 6,
            AlphanumericEncoder: 5,
            ByteEncoder: 5,
            BareByteEncoder: 5,
            KanjiEncoder: 4,
        },
        "codewords_total": 33,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 33,
                    "k": 21,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 33,
                    "k": 11,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 168,
            ErrorCorrectionLevel.H: 88,
        },
    },
    "R9x77": {
        "version_indicator": 0b00111,
        "height": 9,
        "width": 77,
        "remainder_bits": 1,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 6,
            ByteEncoder: 5,
            BareByteEncoder: 5,
            KanjiEncoder: 5,
        },
        "codewords_total": 49,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 49,
                    "k": 31,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 24,
                    "k": 8,
                },
                {
                    "num": 1,
                    "c": 25,
                    "k": 9,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 248,
            ErrorCorrectionLevel.H: 136,
        },
    },
    "R9x99": {
        "version_indicator": 0b01000,
        "height": 9,
        "width": 99,
        "remainder_bits": 4,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 6,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 5,
        },
        "codewords_total": 66,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 66,
                    "k": 42,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 2,
                    "c": 33,
                    "k": 11,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 336,
            ErrorCorrectionLevel.H: 176,
        },
    },
    "R9x139": {
        "version_indicator": 0b01001,
        "height": 9,
        "width": 139,
        "remainder_bits": 5,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 7,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 6,
        },
        "codewords_total": 99,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 49,
                    "k": 31,
                },
                {
                    "num": 1,
                    "c": 50,
                    "k": 32,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 3,
                    "c": 33,
                    "k": 11,
                }
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 504,
            ErrorCorrectionLevel.H: 264,
        },
    },
    "R11x27": {
        "version_indicator": 0b01010,
        "height": 11,
        "width": 27,
        "remainder_bits": 2,
        "character_count_indicator_length": {
            NumericEncoder: 4,
            AlphanumericEncoder: 4,
            ByteEncoder: 3,
            BareByteEncoder: 3,
            KanjiEncoder: 2,
        },
        "codewords_total": 15,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 15,
                    "k": 7,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 15,
                    "k": 5,
                }
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 56,
            ErrorCorrectionLevel.H: 40,
        },
    },
    "R11x43": {
        "version_indicator": 0b01011,
        "height": 11,
        "width": 43,
        "remainder_bits": 1,
        "character_count_indicator_length": {
            NumericEncoder: 6,
            AlphanumericEncoder: 5,
            ByteEncoder: 5,
            BareByteEncoder: 5,
            KanjiEncoder: 4,
        },
        "codewords_total": 31,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 31,
                    "k": 19,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 31,
                    "k": 11,
                }
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 152,
            ErrorCorrectionLevel.H: 88,
        },
    },
    "R11x59": {
        "version_indicator": 0b01100,
        "height": 11,
        "width": 59,
        "remainder_bits": 0,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 6,
            ByteEncoder: 5,
            BareByteEncoder: 5,
            KanjiEncoder: 5,
        },
        "codewords_total": 47,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 47,
                    "k": 31,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 23,
                    "k": 7,
                },
                {
                    "num": 1,
                    "c": 24,
                    "k": 8,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 248,
            ErrorCorrectionLevel.H: 120,
        },
    },
    "R11x77": {
        "version_indicator": 0b01101,
        "height": 11,
        "width": 77,
        "remainder_bits": 2,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 6,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 5,
        },
        "codewords_total": 67,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 67,
                    "k": 43,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 33,
                    "k": 11,
                },
                {
                    "num": 1,
                    "c": 34,
                    "k": 12,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 344,
            ErrorCorrectionLevel.H: 184,
        },
    },
    "R11x99": {
        "version_indicator": 0b01110,
        "height": 11,
        "width": 99,
        "remainder_bits": 7,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 7,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 6,
        },
        "codewords_total": 89,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 44,
                    "k": 28,
                },
                {
                    "num": 1,
                    "c": 45,
                    "k": 29,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 44,
                    "k": 14,
                },
                {
                    "num": 1,
                    "c": 45,
                    "k": 15,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 456,
            ErrorCorrectionLevel.H: 232,
        },
    },
    "R11x139": {
        "version_indicator": 0b01111,
        "height": 11,
        "width": 139,
        "remainder_bits": 6,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 7,
            ByteEncoder: 7,
            BareByteEncoder: 7,
            KanjiEncoder: 6,
        },
        "codewords_total": 132,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 2,
                    "c": 66,
                    "k": 42,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 3,
                    "c": 44,
                    "k": 14,
                }
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 672,
            ErrorCorrectionLevel.H: 336,
        },
    },
    "R13x27": {
        "version_indicator": 0b10000,
        "height": 13,
        "width": 27,
        "character_count_indicator_length": {
            NumericEncoder: 5,
            AlphanumericEncoder: 5,
            ByteEncoder: 4,
            BareByteEncoder: 4,
            KanjiEncoder: 3,
        },
        "remainder_bits": 4,
        "codewords_total": 21,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 21,
                    "k": 14,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 21,
                    "k": 7,
                }
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 96,
            ErrorCorrectionLevel.H: 56,
        },
    },
    "R13x43": {
        "version_indicator": 0b10001,
        "height": 13,
        "width": 43,
        "remainder_bits": 1,
        "character_count_indicator_length": {
            NumericEncoder: 6,
            AlphanumericEncoder: 6,
            ByteEncoder: 5,
            BareByteEncoder: 5,
            KanjiEncoder: 5,
        },
        "codewords_total": 41,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 41,
                    "k": 27,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 41,
                    "k": 13,
                }
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 216,
            ErrorCorrectionLevel.H: 104,
        },
    },
    "R13x59": {
        "version_indicator": 0b10010,
        "height": 13,
        "width": 59,
        "remainder_bits": 6,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 6,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 5,
        },
        "codewords_total": 60,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 60,
                    "k": 38,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 2,
                    "c": 30,
                    "k": 10,
                }
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 304,
            ErrorCorrectionLevel.H: 160,
        },
    },
    "R13x77": {
        "version_indicator": 0b10011,
        "height": 13,
        "width": 77,
        "remainder_bits": 4,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 7,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 6,
        },
        "codewords_total": 85,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 42,
                    "k": 26,
                },
                {
                    "num": 1,
                    "c": 43,
                    "k": 27,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 42,
                    "k": 14,
                },
                {
                    "num": 1,
                    "c": 43,
                    "k": 15,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 424,
            ErrorCorrectionLevel.H: 232,
        },
    },
    "R13x99": {
        "version_indicator": 0b10100,
        "height": 13,
        "width": 99,
        "remainder_bits": 3,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 7,
            ByteEncoder: 7,
            BareByteEncoder: 7,
            KanjiEncoder: 6,
        },
        "codewords_total": 113,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 56,
                    "k": 36,
                },
                {
                    "num": 1,
                    "c": 57,
                    "k": 37,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 37,
                    "k": 11,
                },
                {
                    "num": 2,
                    "c": 38,
                    "k": 12,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 584,
            ErrorCorrectionLevel.H: 280,
        },
    },
    "R13x139": {
        "version_indicator": 0b10101,
        "height": 13,
        "width": 139,
        "remainder_bits": 0,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 8,
            ByteEncoder: 7,
            BareByteEncoder: 7,
            KanjiEncoder: 7,
        },
        "codewords_total": 166,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 2,
                    "c": 55,
                    "k": 35,
                },
                {
                    "num": 1,
                    "c": 56,
                    "k": 36,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 2,
                    "c": 41,
                    "k": 13,
                },
                {
                    "num": 2,
                    "c": 42,
                    "k": 14,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 848,
            ErrorCorrectionLevel.H: 432,
        },
    },
    "R15x43": {
        "version_indicator": 0b10110,
        "height": 15,
        "width": 43,
        "remainder_bits": 1,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 6,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 5,
        },
        "codewords_total": 51,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 51,
                    "k": 33,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 25,
                    "k": 7,
                },
                {
                    "num": 1,
                    "c": 26,
                    "k": 8,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 264,
            ErrorCorrectionLevel.H: 120,
        },
    },
    "R15x59": {
        "version_indicator": 0b10111,
        "height": 15,
        "width": 59,
        "remainder_bits": 4,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 7,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 5,
        },
        "codewords_total": 74,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 74,
                    "k": 48,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 2,
                    "c": 37,
                    "k": 13,
                }
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 384,
            ErrorCorrectionLevel.H: 208,
        },
    },
    "R15x77": {
        "version_indicator": 0b11000,
        "height": 15,
        "width": 77,
        "remainder_bits": 6,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 7,
            ByteEncoder: 7,
            BareByteEncoder: 7,
            KanjiEncoder: 6,
        },
        "codewords_total": 103,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 51,
                    "k": 33,
                },
                {
                    "num": 1,
                    "c": 52,
                    "k": 34,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 2,
                    "c": 34,
                    "k": 10,
                },
                {
                    "num": 1,
                    "c": 35,
                    "k": 11,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 536,
            ErrorCorrectionLevel.H: 248,
        },
    },
    "R15x99": {
        "version_indicator": 0b11001,
        "height": 15,
        "width": 99,
        "remainder_bits": 7,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 7,
            ByteEncoder: 7,
            BareByteEncoder: 7,
            KanjiEncoder: 6,
        },
        "codewords_total": 136,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 2,
                    "c": 68,
                    "k": 44,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 4,
                    "c": 34,
                    "k": 12,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 704,
            ErrorCorrectionLevel.H: 384,
        },
    },
    "R15x139": {
        "version_indicator": 0b11010,
        "height": 15,
        "width": 139,
        "remainder_bits": 2,
        "character_count_indicator_length": {
            NumericEncoder: 9,
            AlphanumericEncoder: 8,
            ByteEncoder: 7,
            BareByteEncoder: 7,
            KanjiEncoder: 7,
        },
        "codewords_total": 199,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 2,
                    "c": 66,
                    "k": 42,
                },
                {
                    "num": 1,
                    "c": 67,
                    "k": 43,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 39,
                    "k": 13,
                },
                {
                    "num": 4,
                    "c": 40,
                    "k": 14,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 1016,
            ErrorCorrectionLevel.H: 552,
        },
    },
    "R17x43": {
        "version_indicator": 0b11011,
        "height": 17,
        "width": 43,
        "remainder_bits": 1,
        "character_count_indicator_length": {
            NumericEncoder: 7,
            AlphanumericEncoder: 6,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 5,
        },
        "codewords_total": 61,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 1,
                    "c": 60,
                    "k": 39,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 30,
                    "k": 10,
                },
                {
                    "num": 1,
                    "c": 31,
                    "k": 11,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 312,
            ErrorCorrectionLevel.H: 168,
        },
    },
    "R17x59": {
        "version_indicator": 0b11100,
        "height": 17,
        "width": 59,
        "remainder_bits": 2,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 7,
            ByteEncoder: 6,
            BareByteEncoder: 6,
            KanjiEncoder: 6,
        },
        "codewords_total": 88,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 2,
                    "c": 44,
                    "k": 28,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 2,
                    "c": 44,
                    "k": 14,
                }
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 448,
            ErrorCorrectionLevel.H: 224,
        },
    },
    "R17x77": {
        "version_indicator": 0b11101,
        "height": 17,
        "width": 77,
        "remainder_bits": 0,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 7,
            ByteEncoder: 7,
            BareByteEncoder: 7,
            KanjiEncoder: 6,
        },
        "codewords_total": 122,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 2,
                    "c": 61,
                    "k": 39,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 1,
                    "c": 40,
                    "k": 12,
                },
                {
                    "num": 2,
                    "c": 41,
                    "k": 13,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 624,
            ErrorCorrectionLevel.H: 304,
        },
    },
    "R17x99": {
        "version_indicator": 0b11110,
        "height": 17,
        "width": 99,
        "remainder_bits": 3,
        "character_count_indicator_length": {
            NumericEncoder: 8,
            AlphanumericEncoder: 8,
            ByteEncoder: 7,
            BareByteEncoder: 7,
            KanjiEncoder: 6,
        },
        "codewords_total": 160,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 2,
                    "c": 53,
                    "k": 33,
                },
                {
                    "num": 1,
                    "c": 54,
                    "k": 34,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 4,
                    "c": 40,
                    "k": 14,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 800,
            ErrorCorrectionLevel.H: 448,
        },
    },
    "R17x139": {
        "version_indicator": 0b11111,
        "height": 17,
        "width": 139,
        "remainder_bits": 4,
        "character_count_indicator_length": {
            NumericEncoder: 9,
            AlphanumericEncoder: 8,
            ByteEncoder: 8,
            BareByteEncoder: 8,
            KanjiEncoder: 7,
        },
        "codewords_total": 232,
        "blocks": {
            ErrorCorrectionLevel.M: [
                {
                    "num": 4,
                    "c": 58,
                    "k": 38,
                },
            ],
            ErrorCorrectionLevel.H: [
                {
                    "num": 2,
                    "c": 38,
                    "k": 12,
                },
                {
                    "num": 4,
                    "c": 39,
                    "k": 13,
                },
            ],
        },
        "number_of_data_bits": {
            ErrorCorrectionLevel.M: 1216,
            ErrorCorrectionLevel.H: 608,
        },
    },
}
