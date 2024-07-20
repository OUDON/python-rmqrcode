from .galois_fields import GaloisFields
from .utilities import msb, to_binary


def compute_bch(data: int):
    data <<= 12
    g = 1 << 12 | 1 << 11 | 1 << 10 | 1 << 9 | 1 << 8 | 1 << 5 | 1 << 2 | 1 << 0

    tmp_data = data
    while msb(tmp_data) >= 13:
        multiple = msb(tmp_data) - 13
        tmp_g = g << multiple
        tmp_data ^= tmp_g
    return tmp_data


gf = GaloisFields()


def compute_reed_solomon(data: list[str], g: list[int], num_error_codewords: int):
    f = list(map(lambda x: int(x, 2), data))

    for i in range(num_error_codewords):
        f.append(0)

    for i in range(len(data)):
        if f[i] == 0:
            continue
        mult = gf.i2e[f[i]]
        for j in range(len(g)):
            f[i + j] ^= gf.e2i[(g[j] + mult) % 255]

    rs_codewords: list[str] = []
    for i in range(num_error_codewords):
        rs_codewords.append(to_binary(f[-num_error_codewords + i], 8))

    return rs_codewords
