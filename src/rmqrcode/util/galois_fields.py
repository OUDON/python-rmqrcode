# GF(2^8)
class GaloisFields:
    e2i: dict[int, int] = {}
    i2e: dict[int, int] = {}

    def __init__(self):
        # Irreducible polynomial in GF(2^8)
        p = (1 << 8) | (1 << 4) | (1 << 3) | (1 << 2) | 1

        self.e2i[0] = 1
        self.e2i[255] = 1
        self.i2e[0] = -1
        self.i2e[1] = 0

        tmp = 1
        for e in range(1, 255):
            tmp <<= 1
            if tmp & (1 << 8):
                tmp ^= p
            self.e2i[e] = tmp
            self.i2e[tmp] = e
