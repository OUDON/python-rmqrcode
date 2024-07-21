from abc import ABC, abstractmethod


class EncoderBase(ABC):
    """An abstract class for encoders"""

    @classmethod
    @abstractmethod
    def mode_indicator(cls) -> str:
        """Mode indicator defined in the Table 2.

        Returns:
            str: Mode indicator like "001".

        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def encode(cls, data: str, character_count_indicator_length: int) -> str:
        """Encodes data and returns it.

        Args:
            data (str): Data to encode.
            character_count_indicator_length: (int): Number of bits of character
                count indicator defined in the Table 3.

        Returns:
            str: Encoded binary as string.

        Raises:
            IllegalCharacterError: If the data includes illegal character.

        """
        if not cls.is_valid_characters(data):
            raise IllegalCharacterError

        res = cls.mode_indicator()
        res += bin(cls.characters_num(data))[2:].zfill(character_count_indicator_length)
        res += cls._encoded_bits(data)
        return res

    @classmethod
    @abstractmethod
    def _encoded_bits(cls, data: str) -> str:
        """Encodes data and returns it.

        This method encodes the raw data without the meta data like the mode
        indicator, the number of data characters and so on.

        Args:
            data (str): Data to encode.

        Returns:
            str: Encoded binary as string.

        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def length(cls, data: str, character_count_indicator_length: int) -> int:
        """Compute the length of the encoded bits.

        Args:
            data (str): Data to encode.

        Returns:
            int: The length of the encoded bits.

        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def characters_num(cls, data: str) -> int:
        """Returns the number of the characters of the data.

        Args:
            data (str): The data to encode.

        Returns:
            int: The number of the characters of the data.

        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def is_valid_characters(cls, data: str) -> bool:
        """Checks whether the data does not include invalid character.

        Args:
            data (str): Data to validate.

        Returns:
            bool: Validation result.

        """
        raise NotImplementedError()


class IllegalCharacterError(ValueError):
    "A class represents an error raised when the given data includes illegal character."
    pass
