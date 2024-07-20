"""A module to make an rMQR Code.

Example:
    Use the rMQR.fit method to make an rMQR automatically with some options.

        qr = rMQR.fit(
            "https://oudon.xyz",
            ecc=ErrorCorrectionLevel.M,
            fit_strategy=FitStrategy.MINIMIZE_WIDTH
        )

    The following example shows how to select the size of an rMQR Code.

        qr = rMQR("R11x139", ErrorCorrectionLevel.H)
        qr.make("https://oudon.xyz")

Attributes:
    QUIET_ZONE_MODULES (int): The width of the quiet zone.

"""

import logging
from typing import TypedDict, Union

from . import encoder
from . import segments as qr_segments
from .enums.color import Color
from .enums.fit_strategy import FitStrategy
from .errors import DataTooLongError, IllegalVersionError, NoSegmentError
from .format.alignment_pattern_coordinates import AlignmentPatternCoordinates
from .format.error_correction_level import ErrorCorrectionLevel
from .format.generator_polynomials import GeneratorPolynomials
from .format.mask import mask
from .format.rmqr_versions import BlockDef, rMQRVersions
from .util.error_correction import compute_bch, compute_reed_solomon
from .util.utilities import split_into_8bits

QUIET_ZONE_MODULES = 2


class _OKVersion(TypedDict):
    version: str
    width: int
    height: int
    segments: list[qr_segments.Segment]


class rMQR:
    """A class to make an rMQR Code."""

    @staticmethod
    def _init_logger():
        """Initializes a logger and returns it.

        Returns:
            logging.RootLogger: Logger

        """
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.DEBUG)
        logger.propagate = True
        return logger

    @staticmethod
    def fit(
        data: str, ecc: ErrorCorrectionLevel = ErrorCorrectionLevel.M, fit_strategy: FitStrategy = FitStrategy.BALANCED
    ):
        """Compute optimized rMQR code with the rMQROptimizer class.

        Args:
            data (str): Data string to encode.
            ecc (rmqrcode.ErrorCorrectionLevel): Error correction level.
            fit_strategy (rmqrcode.FitStrategy): Strategy how determine rMQR Code version.

        Returns:
            rmqrcode.rMQR: Optimized rMQR Code.

        Raises:
            rmqrcode.DataTooLongError: If the data is too long to encode.

        """
        return rMQROptimizer.compute(data, ecc, fit_strategy)

    def _optimized_segments(self, data: str):
        """Returns optimized segments computed by SegmentOptimizer.

        Args:
            data (str): The data to encode.

        Returns:
            list: The list of segments.

        """
        optimizer = qr_segments.SegmentOptimizer()
        return optimizer.compute(data, self.version_name(), self._error_correction_level)

    def __init__(
        self,
        version: str,
        ecc: ErrorCorrectionLevel,
        with_quiet_zone: bool = True,
        logger: Union[logging.Logger, None] = None,
    ):
        self._logger = logger or rMQR._init_logger()

        if not rMQR.validate_version(version):
            raise IllegalVersionError("The rMQR version is illegal.")

        self._version_name = version
        self._qr_version = rMQRVersions[version]
        self._height = self._qr_version["height"]
        self._width = self._qr_version["width"]
        self._error_correction_level = ecc
        self._qr = rMQRCore(self._width, self._height)
        self._segments: list[qr_segments.Segment] = []

    def add_segment(self, data: str, encoder_class: type[encoder.encoder_base.EncoderBase] = encoder.ByteEncoder):
        """Adds the segment.

        A segment consists of data and an encoding mode.

        Args:
            data (str): The data.
            encoder_class (abc.ABCMeta): Pass a subclass of EncoderBase to select encoding mode.
                Using ByteEncoder by default.

        Returns:
            void

        """
        self._segments.append({"data": data, "encoder_class": encoder_class})

    def add_segments(self, segments: list[qr_segments.Segment]):
        """Add the segments.

        Args:
            segments (list): The list of segments.

        Returns:
            void

        """
        for segment in segments:
            self.add_segment(segment["data"], segment["encoder_class"])

    def make(self):
        """Makes an rMQR Code for stored segments.

        This method makes an rMQR Code for stored segments. Before call this,
        you need add segments at least one by the add_segment method.

        Returns:
            void

        Raises:
            NoSegmentError: If no segment are stored.

        """
        if len(self._segments) < 1:
            raise NoSegmentError()
        try:
            encoded_data = self._encode_data()
        except DataTooLongError:
            raise DataTooLongError()

        self._qr.put_finder_patterns()
        self._qr.put_corner_finder_pattern()
        self._qr.put_alignment_pattern()
        self._qr.put_timing_pattern()

        format_information = self._compute_format_info()
        self._qr.put_format_information(format_information)

        codewords_num = self._qr_version["codewords_total"]
        codewords = self._make_codewords(encoded_data, codewords_num)
        blocks = self._split_into_blocks(codewords, self._qr_version["blocks"][self._error_correction_level])
        final_codewords = self._make_final_codewords(blocks)
        self._qr.put_data(final_codewords, self._qr_version["remainder_bits"])

    def _encode_data(self):
        """Encodes the data.

        This method encodes the data for added segments. This method concatenates the
        encoded data of each segments. Finally, this concatenates the terminator if possible.

        Returns:
            str: The encoded data.

        """
        data_bits_max = self._qr_version["number_of_data_bits"][self._error_correction_level]

        res = ""
        for segment in self._segments:
            character_count_indicator_length = self._qr_version["character_count_indicator_length"][
                segment["encoder_class"]
            ]
            res += segment["encoder_class"].encode(segment["data"], character_count_indicator_length)
        res = self._append_terminator_if_possible(res, data_bits_max)

        if len(res) > data_bits_max:
            raise DataTooLongError("The data is too long.")

        return res

    def _append_terminator_if_possible(self, data: str, data_bits_max: int):
        """Appends the terminator.

        This method appends the terminator at the end of data and returns the
        appended string. The terminator shall be omitted if the length of string
        after appending the terminator greater than the rMQR code capacity.

        Args:
            data: The data.
            data_bits_max: The max length of data bits.

        Returns:
            str: The string after appending the terminator.

        """
        if len(data) + 3 <= data_bits_max:
            data += "000"
        return data

    def version_name(self):
        """Returns the version name.

        Returns:
            str: The version name.

        Examples:
            >>> qr.version_name()
                "R13x77"

        """
        return f"R{self._height}x{self._width}"

    def size(self):
        """Returns the size.

        Returns:
            tuple: The rMQR Code size.

        Examples:
            >>> qr.size()
                (77, 13)

        Note:
            This not includes the quiet zone.

        """
        return (self.width(), self.height())

    def height(self):
        """Returns the height.

        Returns:
            int: The height.

        Note:
            This not includes the quiet zone.

        """
        return self._height

    def width(self):
        """Returns the width.

        Returns:
            int: The width.

        Note:
            This not includes the quiet zone.

        """
        return self._width

    def to_list(self, with_quiet_zone: bool = True):
        """Converts to two-dimensional list and returns it.

        The value is 1 for the dark module and 0 for the light module.

        Args:
            with_quiet_zone (bool): Flag to select whether include the quiet zone.

        Returns:
            list: Converted list.

        """
        res: list[list[int]] = []
        if with_quiet_zone:
            for y in range(QUIET_ZONE_MODULES):
                res.append([0] * (self.width() + QUIET_ZONE_MODULES * 2))
            for row in self._qr.to_binary_list():
                res.append([0] * QUIET_ZONE_MODULES + row + [0] * QUIET_ZONE_MODULES)
            for y in range(QUIET_ZONE_MODULES):
                res.append([0] * (self.width() + QUIET_ZONE_MODULES * 2))
        else:
            res = self._qr.to_binary_list()
        return res

    def __str__(self, with_quiet_zone: bool = True):
        res = f"rMQR Version R{self._height}x{self._width}:\n"
        res += self._qr.__str__(with_quiet_zone)
        return res

    def _compute_format_info(self):
        """Computes format information with BCH code."""
        format_information_data = self._qr_version["version_indicator"]
        if self._error_correction_level == ErrorCorrectionLevel.H:
            format_information_data |= 1 << 5
        reminder_polynomial = compute_bch(format_information_data)
        format_information_data = format_information_data << 12 | reminder_polynomial
        return format_information_data

    def _make_codewords(self, encoded_data: str, codewords_num: int):
        """Makes codeword sequence from encoded data.

        If the length of generated codeword sequence is less than the `codewords_num`,
        appends the reminder codewords 11101100 and 00010001 alternately to meet the
        requirements of number of codewords.

        Args:
            encoded_data (str): The encoded data.
            codewords_num (int): The number of codewords.

        Returns:
            list: The list of codeword strings.

        """
        codewords = split_into_8bits(encoded_data)
        while True:
            if len(codewords) >= codewords_num:
                break
            codewords.append("11101100")
            if len(codewords) >= codewords_num:
                break
            codewords.append("00010001")
        return codewords

    def _split_into_blocks(self, codewords: list[str], blocks_definition: list[BlockDef]):
        """Splits codewords into several blocks.

        Args:
            codewords (list): The list of codeword strings.
            blocks_definition: The list of dict.

        Returns:
            list: The list of Block object.

        """
        data_idx = 0
        blocks: list[Block] = []
        for block_definition in blocks_definition:
            for i in range(block_definition["num"]):
                data_codewords_num = block_definition["k"]
                ecc_codewords_num = block_definition["c"] - block_definition["k"]
                codewords_in_block = codewords[data_idx : data_idx + data_codewords_num]
                block = Block(data_codewords_num, ecc_codewords_num)
                block.set_data_and_compute_ecc(codewords_in_block)
                blocks.append(block)
                data_idx += data_codewords_num
        return blocks

    def _make_final_codewords(self, blocks: list["Block"]):
        """Makes the final message codeword sequence.

        This method computes the final codeword sequence from the given blocks. For example,
        we consider the following blocks. The blocks consists of three blocks. Block1 contains
        two data blocks and three ecc blocks. Block2 contains three data blocks and three ecc blocks.
        Block3 contains three data blocks and three ecc blocks.

            Block1: Data#1 Data#2 ------ Ecc#1 Ecc#2 Ecc#3
            Block2: Data#3 Data#4 Data#5 Ecc#4 Ecc#5 Ecc#6
            Block3: Data#6 Data#7 Data#8 Ecc#7 Ecc#8 Ecc#9

        The final codeword sequence for this example is placed in the following order.

            [Data#1, Data#3, Data#6, Data#2, Data#4, Data#7, Data#5, Data#8,
                Ecc#1, Ecc#4, Ecc#7, Ecc#2, Ecc#5, Ecc#8, Ecc#3, Ecc#6, Ecc#9]

        Args:
            blocks (list): The list of Block objects.

        Returns:
            list: The list of codeword strings.

        """
        final_codewords: list[str] = []
        # Add data codewords
        # The last block always has the most codewords.
        for i in range(blocks[-1].data_length()):
            for block in blocks:
                try:
                    data_codeword = block.get_data_at(i)
                except IndexError:
                    break
                else:
                    final_codewords.append(data_codeword)
                    self._logger.debug(f"Put QR data codeword {i} : {data_codeword}")

        # Add ecc codewords
        # The last block always has the most codewords.
        for i in range(blocks[-1].ecc_length()):
            for block in blocks:
                try:
                    ecc_codeword = block.get_ecc_at(i)
                except IndexError:
                    break
                else:
                    final_codewords.append(ecc_codeword)
                    self._logger.debug(f"Put RS data codewords {i} : {ecc_codeword}")
        return final_codewords

    @staticmethod
    def validate_version(version_name: str):
        """Check if the given version_name is valid

        Args:
            version_name (str): Version name.

        Returns:
            bool: Validation result.

        Example:
            >>> rMQR.validate_version("R13x77")
                True

            >>> rMQR.validate_version("R14x55")
                False

            >>> rMQR.validate_version("13, 77")
                False

        """
        return version_name in rMQRVersions


class rMQROptimizer:
    """A class to compute optimized rMQR code for input data."""

    @staticmethod
    def compute(data: str, ecc: ErrorCorrectionLevel, fit_strategy: FitStrategy):
        """Attempts to make an rMQR have optimized version for given data.

        Args:
            data (str): Data string to encode.
            ecc (rmqrcode.ErrorCorrectionLevel): Error correction level.
            fit_strategy (rmqrcode.FitStrategy): Strategy how determine rMQR Code version.

        Returns:
            rmqrcode.rMQR: Optimized rMQR Code.

        Raises:
            rmqrcode.DataTooLongError: If the data is too long to encode.

        """
        ok_versions: list[_OKVersion] = []
        determined_width: set[int] = set()
        determined_height: set[int] = set()

        for version_name, qr_version in rMQRVersions.items():
            optimizer = qr_segments.SegmentOptimizer()
            try:
                optimized_segments = optimizer.compute(data, version_name, ecc)
            except DataTooLongError:
                continue

            width, height = qr_version["width"], qr_version["height"]
            if width not in determined_width and height not in determined_height:
                determined_width.add(width)
                determined_height.add(height)
                ok_versions.append(
                    {
                        "version": version_name,
                        "width": width,
                        "height": height,
                        "segments": optimized_segments,
                    }
                )

        if len(ok_versions) == 0:
            raise DataTooLongError("The data is too long.")

        if fit_strategy == FitStrategy.MINIMIZE_WIDTH:

            def sort_key(x: _OKVersion):
                return x["width"]

        elif fit_strategy == FitStrategy.MINIMIZE_HEIGHT:

            def sort_key(x: _OKVersion):
                return x["height"]

        elif fit_strategy == FitStrategy.BALANCED:

            def sort_key(x: _OKVersion):
                return x["height"] * 9 + x["width"]

        selected = sorted(ok_versions, key=sort_key)[0]
        qr = rMQR(selected["version"], ecc)
        qr.add_segments(selected["segments"])
        qr.make()
        return qr


class rMQRCore:
    "A class correspond to a grid of modules of rMQR code."

    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._qr = [[Color.UNDEFINED for x in range(self._width)] for y in range(self._height)]

    def get_data(self, x: int, y: int):
        """Returns the module value at x-th column and y-th row.

        Args:
            x (int): The index of x.
            y (int): The index of y.

        Returns:
            rmqrcode.Color: The module value.

        Raises:
            IndexError: If the index is invalid.

        """
        if 0 > x or self._width < x:
            raise IndexError("x index out of range")
        if 0 > y or self._height < y:
            raise IndexError("y index out of range")
        return self._qr[y][x]

    def to_binary_list(self):
        """Converts to two-dimensional list and returns it.

        The value is 1 for the dark module and 0 for the light module.

        Args:
            with_quiet_zone (bool): Flag to select whether include the quiet zone.

        Returns:
            list: Converted list.

        Note:
            This not includes the quiet zone.
        """
        return [list(map(lambda x: 1 if x == Color.BLACK else 0, column)) for column in self._qr]

    def put_finder_patterns(self):
        self._put_finder_pattern()
        self._put_finder_sub_pattern()

    def _put_finder_pattern(self):
        # Outer square
        for i in range(7):
            for j in range(7):
                if i == 0 or i == 6 or j == 0 or j == 6:
                    self._qr[i][j] = Color.BLACK
                else:
                    self._qr[i][j] = Color.WHITE

        # Inner square
        for i in range(3):
            for j in range(3):
                self._qr[2 + i][2 + j] = Color.BLACK

        # Separator
        for n in range(8):
            if n < self._height:
                self._qr[n][7] = Color.WHITE

            if self._height >= 9:
                self._qr[7][n] = Color.WHITE

    def _put_finder_sub_pattern(self):
        # Outer square
        for i in range(5):
            for j in range(5):
                color = Color.BLACK if i == 0 or i == 4 or j == 0 or j == 4 else Color.WHITE
                self._qr[self._height - i - 1][self._width - j - 1] = color

        # Inner square
        self._qr[self._height - 1 - 2][self._width - 1 - 2] = Color.BLACK

    def put_corner_finder_pattern(self):
        # Corner finder pattern
        # Bottom left
        self._qr[self._height - 1][0] = Color.BLACK
        self._qr[self._height - 1][1] = Color.BLACK
        self._qr[self._height - 1][2] = Color.BLACK

        if self._height >= 11:
            self._qr[self._height - 2][0] = Color.BLACK
            self._qr[self._height - 2][1] = Color.WHITE

        # Top right
        self._qr[0][self._width - 1] = Color.BLACK
        self._qr[0][self._width - 2] = Color.BLACK
        self._qr[1][self._width - 1] = Color.BLACK
        self._qr[1][self._width - 2] = Color.WHITE

    def put_alignment_pattern(self):
        center_xs = AlignmentPatternCoordinates[self._width]
        for center_x in center_xs:
            for i in range(3):
                for j in range(3):
                    color = Color.BLACK if i == 0 or i == 2 or j == 0 or j == 2 else Color.WHITE
                    # Top side
                    self._qr[i][center_x + j - 1] = color
                    # Bottom side
                    self._qr[self._height - 1 - i][center_x + j - 1] = color

    def put_timing_pattern(self):
        self._put_timing_pattern_horizontal()
        self._put_timing_pattern_vertical()

    def _put_timing_pattern_horizontal(self):
        for j in range(self._width):
            color = Color.BLACK if (j + 1) % 2 else Color.WHITE
            for i in [0, self._height - 1]:
                if self._qr[i][j] == Color.UNDEFINED:
                    self._qr[i][j] = color

    def _put_timing_pattern_vertical(self):
        center_xs = [0, self._width - 1]
        center_xs.extend(AlignmentPatternCoordinates[self._width])
        for i in range(self._height):
            color = Color.BLACK if (i + 1) % 2 else Color.WHITE
            for j in center_xs:
                if self._qr[i][j] == Color.UNDEFINED:
                    self._qr[i][j] = color

    def put_format_information(self, format_information: int):
        """Format information placement.

        Args:
            format_information (int): The format information.

        Returns:
            void

        """
        self._put_format_information_finder_pattern_side(format_information)
        self._put_format_information_finder_sub_pattern_side(format_information)

    def _put_format_information_finder_pattern_side(self, format_information: int):
        """Format information placement (finder pattern side).

        This method computes masked format information data and puts it. The mask
        pattern is 011111101010110010.

        Args:
            format_information (int): The format information.

        Returns:
            void

        """
        mask = 0b011111101010110010
        format_information ^= mask

        si, sj = 1, 8
        for n in range(18):
            di = n % 5
            dj = n // 5
            self._qr[si + di][sj + dj] = Color.BLACK if format_information >> n & 1 else Color.WHITE

    def _put_format_information_finder_sub_pattern_side(self, format_information: int):
        """Format information placement (finder sub pattern side).

        This method computes masked format information data and puts it. The mask
        pattern is 100000101001111011.

        Args:
            format_information (int): The format information.

        Returns:
            void

        """
        mask = 0b100000101001111011
        format_information ^= mask

        si, sj = self._height - 1 - 5, self._width - 1 - 7
        for n in range(15):
            di = n % 5
            dj = n // 5
            self._qr[si + di][sj + dj] = Color.BLACK if format_information >> n & 1 else Color.WHITE
        self._qr[self._height - 1 - 5][self._width - 1 - 4] = (
            Color.BLACK if format_information >> 15 & 1 else Color.WHITE
        )
        self._qr[self._height - 1 - 5][self._width - 1 - 3] = (
            Color.BLACK if format_information >> 16 & 1 else Color.WHITE
        )
        self._qr[self._height - 1 - 5][self._width - 1 - 2] = (
            Color.BLACK if format_information >> 17 & 1 else Color.WHITE
        )

    def put_data(self, final_codewords: list[str], remainder_bits_num: int):
        """Symbol character placement.

        This method puts data into the encoding region of the rMQR Code. The data
        should be encoded by NumericEncoder, AlphanumericEncoder, ByteEncoder or KanjiEncoder.
        Also this method computes a two-dimensional list shows where encoding region at the
        same time. And returns the list.
        See: "7.7.3 Symbol character placement" in the ISO/IEC 23941.

        Args:
            encoded_data (str): The data after encoding. Expected all segments are joined.
            reminder_bits_num (int): The number of modules without data.

        Returns:
            list: A two-dimensional list shows where encoding region.

        """
        mask_area = self._put_final_codewords(final_codewords, remainder_bits_num)
        self._apply_mask(mask_area)

    def _put_final_codewords(self, final_codewords: list[str], reminder_bits_num: int):
        """Puts the final codeword sequence.

        This method puts the final codeword sequence into the encoding region of the rMQR Code.
        The `final_codewords` is computed by self._make_final_codewords method. Also, this method
        computes a two-dimensional list shows where encoding region at the same time.
        And returns the list.

        Args:
            final_codewords (list): The list of the final codeword strings.
            reminder_bits_num (int): The number of modules without data.

        Returns:
            list: A two-dimensional list shows where encoding region.

        """
        dy = -1  # Up
        current_codeword_idx = 0
        current_bit_idx = 0
        cx, cy = self._width - 2, self._height - 6
        remaining_remainder_bits = reminder_bits_num
        mask_area = [[False for i in range(self._width)] for j in range(self._height)]

        while True:
            for x in [cx, cx - 1]:
                if self._qr[cy][x] == Color.UNDEFINED:
                    # Process only empty cell
                    if current_codeword_idx == len(final_codewords):
                        # Remainder bits
                        self._qr[cy][x] = Color.WHITE
                        mask_area[cy][x] = True
                        remaining_remainder_bits -= 1
                    else:
                        # Codewords
                        self._qr[cy][x] = (
                            Color.BLACK
                            if final_codewords[current_codeword_idx][current_bit_idx] == "1"
                            else Color.WHITE
                        )
                        mask_area[cy][x] = True
                        current_bit_idx += 1
                        if current_bit_idx == 8:
                            current_bit_idx = 0
                            current_codeword_idx += 1

                    if current_codeword_idx == len(final_codewords) and remaining_remainder_bits == 0:
                        break

            if current_codeword_idx == len(final_codewords) and remaining_remainder_bits == 0:
                break

            # Update current coordinates
            if dy < 0 and cy == 1:
                cx -= 2
                dy = 1
            elif dy > 0 and cy == self._height - 1 - 1:
                cx -= 2
                dy = -1
            else:
                cy += dy

        return mask_area

    def _apply_mask(self, mask_area: list[list[bool]]):
        """Data masking.

        This method applies the data mask.

        Args:
            mask_area (list): A two-dimensional list shows where encoding region.
                This is computed by self._put_data().

        Returns:
            void

        """
        for y in range(self._height):
            for x in range(self._width):
                if not mask_area[y][x]:
                    continue
                if mask(x, y):
                    if self._qr[y][x] == Color.BLACK:
                        self._qr[y][x] = Color.WHITE
                    elif self._qr[y][x] == Color.WHITE:
                        self._qr[y][x] = Color.BLACK

    def __str__(self, with_quiet_zone: bool = True):
        show = {
            Color.WHITE: "_",
            Color.BLACK: "X",
            Color.UNDEFINED: "?",
        }

        res = ""
        if with_quiet_zone:
            res += (show[Color.WHITE] * (self._width + QUIET_ZONE_MODULES * 2) + "\n") * QUIET_ZONE_MODULES

        for y in range(self._height):
            if with_quiet_zone:
                res += show[Color.WHITE] * QUIET_ZONE_MODULES

            for x in range(self._width):
                res += show[self._qr[y][x]]

            if with_quiet_zone:
                res += show[Color.WHITE] * QUIET_ZONE_MODULES
            res += "\n"

        if with_quiet_zone:
            res += (show[Color.WHITE] * (self._width + QUIET_ZONE_MODULES * 2) + "\n") * QUIET_ZONE_MODULES
        return res


class Block:
    """A class represents data block.

    This class represents data block. A block consists data part and error correction
    code (ecc) part.

    """

    def __init__(self, data_codewords_num: int, ecc_codewords_num: int):
        self._data_codewords_num = data_codewords_num
        self._data_codewords: list[str] = []
        self._ecc_codewords_num = ecc_codewords_num
        self._ecc_codewords: list[str] = []

    def set_data_and_compute_ecc(self, data_codewords: list[str]):
        """Set data and compute ecc.

        Args:
            data_codewords (list): The list of codeword strings.

        Returns:
            void

        """
        self._data_codewords = data_codewords
        self._compute_ecc_codewords()

    def get_data_at(self, index: int):
        """Get data codeword at the index.

        Args:
            index (int): The index.

        Return:
            str: The data codeword.

        """
        return self._data_codewords[index]

    def get_ecc_at(self, index: int):
        """Get ecc codeword at the index.

        Args:
            index (int): The index.

        Return:
            str: The ecc codeword.

        """
        return self._ecc_codewords[index]

    def data_length(self):
        """Get the number of data codewords"""
        return len(self._data_codewords)

    def ecc_length(self):
        """Get the number of ecc codewords"""
        return len(self._ecc_codewords)

    def _compute_ecc_codewords(self):
        """Computes the ecc codewords with the data codewords."""
        g = GeneratorPolynomials[self._ecc_codewords_num]
        self._ecc_codewords = compute_reed_solomon(self._data_codewords, g, self._ecc_codewords_num)
