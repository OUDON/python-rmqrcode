from typing import TypedDict

from . import encoder
from .errors import DataTooLongError
from .format.error_correction_level import ErrorCorrectionLevel
from .format.rmqr_versions import rMQRVersions


class Best(TypedDict):
    cost: int
    index: tuple[int, int, int]
    """n, mode, unfilled_length"""


class Segment(TypedDict):
    data: str
    encoder_class: type[encoder.encoder_base.EncoderBase]


encoders: list[type[encoder.encoder_base.EncoderBase]] = [
    encoder.NumericEncoder,
    encoder.AlphanumericEncoder,
    encoder.ByteEncoder,
    encoder.KanjiEncoder,
]


def compute_length(segments: list[Segment], version_name: str):
    """Computes the sum of length of the segments.

    Args:
        segments (list): The list of segment.
        version_name (str): The version name.

    Returns:
        int: The sum of the length of the segments.

    """
    return sum(
        map(
            lambda s: s["encoder_class"].length(
                s["data"], rMQRVersions[version_name]["character_count_indicator_length"][s["encoder_class"]]
            ),
            segments,
        )
    )


class SegmentOptimizer:
    """A class for computing optimal segmentation of the given data by dynamic programming.

    Attributes:
        MAX_CHARACTER (int): The maximum characters of the given data.
        INF (int): Large enough value. This is used as initial value of the dynamic programming table.

    """

    MAX_CHARACTER = 360
    INF = 100000

    def __init__(self):
        self.dp: list[list[list[int]]] = [
            [[self.INF for n in range(3)] for mode in range(4)] for length in range(self.MAX_CHARACTER + 1)
        ]
        self.parents: list[list[list[tuple[int, int, int]]]] = [
            [[(-1, -1, -1) for n in range(3)] for mode in range(4)] for length in range(self.MAX_CHARACTER + 1)
        ]

    def compute(self, data: str, version: str, ecc: ErrorCorrectionLevel):
        """Computes the optimize segmentation for the given data.

        Args:
            data (str): The data to encode.
            version (str): The version name.
            ecc (rmqrcode.ErrorCorrectionLevel): The error correction level.

        Returns:
            list: The list of segments.

        Raises:
            rmqrcode.DataTooLongError: If the data is too long to encode.

        """
        if len(data) > self.MAX_CHARACTER:
            raise DataTooLongError()

        self.qr_version = rMQRVersions[version]
        self._compute_costs(data)
        best = self._find_best(data)
        if best["cost"] > self.qr_version["number_of_data_bits"][ecc]:
            raise DataTooLongError

        path = self._reconstruct_path(best["index"])
        segments = self._compute_segments(path, data)
        return segments

    def _compute_costs(self, data: str):
        """Computes costs by dynamic programming.

        This method computes costs of the dynamic programming table. Define
        dp[n][mode][unfilled_length] as the minimize bit length when encode only
        the `n`-th leading characters which the last character is encoded in `mode`
        and the remainder bits length is `unfilled_length`.

        Args:
            data (str): The data to encode.

        Returns:
            void

        """
        for mode in range(len(encoders)):
            encoder_class = encoders[mode]
            character_count_indicator_length = self.qr_version["character_count_indicator_length"][encoder_class]
            self.dp[0][mode][0] = encoder_class.length("", character_count_indicator_length)
            self.parents[0][mode][0] = (0, 0, 0)

        for n in range(0, len(data)):
            for mode in range(4):
                for unfilled_length in range(3):
                    if self.dp[n][mode][unfilled_length] == self.INF:
                        continue

                    for new_mode in range(4):
                        if not encoders[new_mode].is_valid_characters(data[n]):
                            continue

                        if new_mode == mode:
                            cost, new_length = self._compute_new_state_without_mode_changing(
                                data[n], new_mode, unfilled_length
                            )
                        else:
                            cost, new_length = self._compute_new_state_with_mode_changing(
                                data[n], new_mode, unfilled_length
                            )

                        if self.dp[n][mode][unfilled_length] + cost < self.dp[n + 1][new_mode][new_length]:
                            self.dp[n + 1][new_mode][new_length] = self.dp[n][mode][unfilled_length] + cost
                            self.parents[n + 1][new_mode][new_length] = (n, mode, unfilled_length)

    def _compute_new_state_without_mode_changing(self, character: str, new_mode: int, unfilled_length: int):
        """Computes the new state values without mode changing.

        Args:
            character (str): The current character. Assume this as one length string.
            new_mode (int): The state of the new mode.
            unfilled_length (int): The state of the current unfilled_length.

        Returns:
            tuple: (cost, new_length).

        """
        encoder_class = encoders[new_mode]
        if encoder_class == encoder.NumericEncoder:
            new_length = (unfilled_length + 1) % 3
            cost = 4 if unfilled_length == 0 else 3
        elif encoder_class == encoder.AlphanumericEncoder:
            new_length = (unfilled_length + 1) % 2
            cost = 6 if unfilled_length == 0 else 5
        elif encoder_class == encoder.ByteEncoder:
            new_length = 0
            cost = 8 * len(character.encode("utf-8"))
        elif encoder_class == encoder.KanjiEncoder:
            new_length = 0
            cost = 13
        else:
            raise NotImplementedError()
        return (cost, new_length)

    def _compute_new_state_with_mode_changing(self, character: str, new_mode: int, unfilled_length: int):
        """Computes the new state values with mode changing.

        Args:
            character (str): The current character. Assume this as one length string.
            new_mode (int): The state of the new mode.
            unfilled_length (int): The state of the current unfilled_length.

        Returns:
            tuple: (cost, new_length).

        """
        encoder_class = encoders[new_mode]
        character_count_indicator_length = self.qr_version["character_count_indicator_length"][encoder_class]
        if encoder_class in [encoder.NumericEncoder, encoder.AlphanumericEncoder]:
            new_length = 1
        elif encoder_class in [encoder.ByteEncoder, encoder.KanjiEncoder]:
            new_length = 0
        else:
            raise NotImplementedError()
        cost = encoder_class.length(character, character_count_indicator_length)
        return (cost, new_length)

    def _find_best(self, data: str) -> Best:
        """Find the index which has the minimum costs.

        Args:
            data (str): The data to encode.

        Returns:
            dict: The dict object includes "cost" and "index". The "cost" is the value of minimum cost.
                The "index" is the index of the dp table as a tuple (n, mode, unfilled_length).

        """
        best = self.INF
        best_index: tuple[int, int, int] = (-1, -1, -1)
        for mode in range(4):
            for unfilled_length in range(3):
                if self.dp[len(data)][mode][unfilled_length] < best:
                    best = self.dp[len(data)][mode][unfilled_length]
                    best_index = (len(data), mode, unfilled_length)
        return {"cost": best, "index": best_index}

    def _reconstruct_path(self, best_index: tuple[int, int, int]):
        """Reconstructs the path.

        Args:
            best_index: The best index computed by self._find_best().

        Returns:
            list: The path of minimum cost in the dynamic programming table.

        """
        path: list[tuple[int, int, int]] = []
        index = best_index
        while index[0] != 0:
            path.append(index)
            index = self.parents[index[0]][index[1]][index[2]]
        path.reverse()
        return path

    def _compute_segments(self, path: list[tuple[int, int, int]], data: str):
        """Computes the segments.

        This method computes the segments. The adjacent characters has same mode are merged.

        Args:
            path (list): The path computed by self._reconstruct_path().
            data (str): The data to encode.

        Returns:
            list: The list of segments.

        """
        segments: list[Segment] = []
        current_segment_data = ""
        current_mode = -1
        for p in path:
            if current_mode == -1:
                current_mode = p[1]
                current_segment_data += data[p[0] - 1]
            elif current_mode == p[1]:
                current_segment_data += data[p[0] - 1]
            else:
                segments.append({"data": current_segment_data, "encoder_class": encoders[current_mode]})
                current_segment_data = data[p[0] - 1]
                current_mode = p[1]
        if current_mode != -1:
            segments.append({"data": current_segment_data, "encoder_class": encoders[current_mode]})
        return segments
