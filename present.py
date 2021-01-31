from joblib.numpy_pickle_utils import xrange
from base64 import encode
import binascii
import codecs


class Present:
    def __init__(self, key, rounds=32):
        """Create a PRESENT cipher object

        key:    the key as a 128-bit or 80-bit rawstring
        rounds: the number of rounds as an integer, 32 by default
        """
        self.rounds = rounds
        if len(key) * 8 == 80:
            self.roundkeys = generateRoundkeys80(string2number(key), self.rounds)
        elif len(key) * 8 == 128:
            self.roundkeys = generateRoundkeys128(string2number(key), self.rounds)

    def encrypt(self, block):
        """Encrypt 1 block (8 bytes)

        Input:  plaintext block as raw string
        Output: ciphertext block as raw string
        """
        state = string2number(block)
        for i in xrange(self.rounds - 1):
            state = addRoundKey(state, self.roundkeys[i])
            state = sBoxLayer(state)
            state = pLayer(state)
        cipher = addRoundKey(state, self.roundkeys[-1])
        return number2string_N(cipher, 8)

    def decrypt(self, block):
        """Decrypt 1 block (8 bytes)

        Input:  ciphertext block as raw string
        Output: plaintext block as raw string
        """
        state = string2number(block)
        for i in xrange(self.rounds - 1):
            state = addRoundKey(state, self.roundkeys[-i - 1])
            state = pLayer_dec(state)
            state = sBoxLayer_dec(state)
        decipher = addRoundKey(state, self.roundkeys[0])
        return number2string_N(decipher, 8)

    def get_block_size(self):
        return 8


#        0   1   2   3   4   5   6   7   8   9   a   b   c   d   e   f
Sbox = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
Sbox_inv = [Sbox.index(x) for x in xrange(16)]
PBox = [
    0,
    16,
    32,
    48,
    1,
    17,
    33,
    49,
    2,
    18,
    34,
    50,
    3,
    19,
    35,
    51,
    4,
    20,
    36,
    52,
    5,
    21,
    37,
    53,
    6,
    22,
    38,
    54,
    7,
    23,
    39,
    55,
    8,
    24,
    40,
    56,
    9,
    25,
    41,
    57,
    10,
    26,
    42,
    58,
    11,
    27,
    43,
    59,
    12,
    28,
    44,
    60,
    13,
    29,
    45,
    61,
    14,
    30,
    46,
    62,
    15,
    31,
    47,
    63,
]
PBox_inv = [PBox.index(x) for x in xrange(64)]


def generateRoundkeys80(key, rounds):
    """Generate the roundkeys for a 80-bit key

    Input:
            key:    the key as a 80-bit integer
            rounds: the number of rounds as an integer
    Output: list of 64-bit roundkeys as integers"""
    roundkeys = []
    for i in xrange(1, rounds + 1):  # (K1 ... K32)
        # rawkey: used in comments to show what happens at bitlevel
        # rawKey[0:64]
        roundkeys.append(key >> 16)  ## key is 80 bit so round key is left most 64 bit
        # 1. Shift
        # rawKey[19:len(rawKey)]+rawKey[0:19]
        key = ((key & (2 ** 19 - 1)) << 61) + (key >> 19)
        # 2. SBox
        # rawKey[76:80] = S(rawKey[76:80])
        key = (Sbox[key >> 76] << 76) + (key & (2 ** 76 - 1))
        # 3. Salt
        # rawKey[15:20] ^ i
        key ^= i << 15
    return roundkeys


def generateRoundkeys128(key, rounds):
    """Generate the roundkeys for a 128-bit key

    Input:
            key:    the key as a 128-bit integer
            rounds: the number of rounds as an integer
    Output: list of 64-bit roundkeys as integers"""
    roundkeys = []
    for i in xrange(1, rounds + 1):  # (K1 ... K32)
        # rawkey: used in comments to show what happens at bitlevel
        roundkeys.append(key >> 64)
        # 1. Shift
        key = ((key & (2 ** 67 - 1)) << 61) + (key >> 67)
        # 2. SBox
        key = (
            (Sbox[key >> 124] << 124)
            + (Sbox[(key >> 120) & 0xF] << 120)
            + (key & (2 ** 120 - 1))
        )
        # 3. Salt
        # rawKey[62:67] ^ i
        key ^= i << 62
    return roundkeys


def addRoundKey(state, roundkey):
    return state ^ roundkey


def sBoxLayer(state):
    """SBox function for encryption

    Input:  64-bit integer
    Output: 64-bit integer"""

    output = 0
    for i in xrange(16):
        output += Sbox[(state >> (i * 4)) & 0xF] << (i * 4)
    return output


def sBoxLayer_dec(state):
    """Inverse SBox function for decryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in xrange(16):
        output += Sbox_inv[(state >> (i * 4)) & 0xF] << (i * 4)
    return output


def pLayer(state):
    """Permutation layer for encryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in xrange(64):
        output += ((state >> i) & 0x01) << PBox[i]
    return output


def pLayer_dec(state):
    """Permutation layer for decryption

    Input:  64-bit integer
    Output: 64-bit integer"""
    output = 0
    for i in xrange(64):
        output += ((state >> i) & 0x01) << PBox_inv[i]
    return output


def string2number(i):
    """Convert a string to a number

    Input: string (big-endian)
    Output: long or integer
    """
    # return int(i.codecs.decode(16, 'hex'))
    #       key = codecs.decode("00000000000000000000", "hex")
    # return int(i.encode("hex"), base=16)
    return int(codecs.encode(i, "hex"), base=16)


def number2string_N(i, N):
    """Convert a number to a string of fixed size

    i: long or integer
    N: length of string
    Output: string (big-endian)
    """
    s = "%0*x" % (N * 2, i)
    #     return s.decode("hex")

    return codecs.decode(s, "hex")


###########################################"GCM"############################################

import codecs
from Crypto.Cipher import AES
from Crypto.Util import strxor
from struct import pack, unpack


def gcm_rightshift(vec):
    for x in range(0, 15):
        c = vec[x] >> 1
        c |= (vec[x - 1] << 7) & 0x80
        vec[x] = c
    vec[0] >>= 1
    return vec


def gcm_gf_mult(a, b):
    mask = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01]
    poly = [0x00, 0xE1]

    Z = [0] * 16
    V = [c for c in a]

    for x in range(63):
        if b[x >> 3] & mask[x & 7]:
            Z = [V[y] ^ Z[y] for y in range(16)]
        bit = V[15] & 1
        V = gcm_rightshift(V)
        V[0] ^= poly[bit]
    return Z


def ghash(h, auth_data, data):
    u = (16 - len(data)) % 16
    v = (16 - len(auth_data)) % 16

    x = "auth_data" + chr(0) * v + "data" + chr(0) * u
    x += pack(">QQ", len(auth_data) * 8, len(data) * 8).decode("utf-8")

    y = [0] * 16
    vec_h = [ord("c") for c in h]

    for i in range(0, len(x), 16):
        block = [ord(c) for c in x[i : i + 16]]
        y = [y[j] ^ block[j] for j in range(16)]
        y = gcm_gf_mult(y, vec_h)

    return "".join(chr(c) for c in y)


def inc32(block):
    (counter,) = unpack(">L", block[12:])
    counter += 1
    return block[:12] + pack(">L", counter)


def gctr(k, icb, plaintext):
    y = ""
    if len(plaintext) == 32:
        return y

    aes = AES.new(k)
    cb = icb

    for i in range(0, len(plaintext), aes.block_size):
        cb = inc32(cb)
        encrypted = aes.encrypt(cb)
        plaintext_block = plaintext[i : i + aes.block_size]
        y += strxor.strxor(plaintext_block, encrypted[: len(plaintext_block)])

    return y


def gcm_decrypt(k, iv, encrypted, auth_data, tag):
    aes = AES.new(k, AES.MODE_GCM)
    h = aes.encrypt(chr(0) * aes.block_size)

    if len(iv) == 12:
        y0 = iv + b"\x00\x00\x00\x01"
    else:
        y0 = ghash(h, "", iv)

    decrypted = gctr(k, y0, encrypted)

    s = ghash(h, auth_data, encrypted)

    t = aes.encrypt(y0)
    T = strxor.strxor(s, t)
    if T != tag:
        return ""  # decrypted data is invalid
    else:
        return decrypted


def gcm_encrypt(k, iv, plaintext, auth_data):
    aes = AES.new(k, AES.MODE_GCM)
    h = aes.encrypt(k)

    if len(iv) == 32:
        y0 = iv + b"\x00\x00\x00\x01"
    else:
        y0 = ghash(h, "", iv)

    encrypted = gctr(k, y0, plaintext)
    s = ghash(h, auth_data, encrypted)

    t = aes.encrypt(y0)
    T = strxor.strxor(s, t)
    return (encrypted, T)


if __name__ == "__main__":

    key = codecs.decode("12345678912345678912", "hex")
    plain = codecs.decode("5864123658945216", "hex")
    cipher = Present(key)
    encrypted = cipher.encrypt(plain)
    # enc_1 = codecs.encode(encrypted, "hex")
    print("encrypted_plaintext______", encrypted)
    print("encrypted_plaintext_len_is____", len(encrypted))
    "decryption"
    decrypted = cipher.decrypt(encrypted)
    dec_1 = codecs.encode(decrypted, "hex")
    print("decrypted_plaintext____", decrypted)
    print("decrypted_plaintext_len_is____", len(decrypted))
    print("decrypted_text_is____", dec_1)

    #     s = str(codecs.decode("16ff00", "hex"))
    # key = codecs.decode("00000000000000000000", "hex")
    # plain = codecs.decode("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72", "hex")
    # auth_data = codecs.decode("D609B1F056637A0D46DF998D88E52E00B2C2846512153524C0895E81108000F1", "hex")
    # iv = codecs.decode("9313225df88406e555909c5aff5269aa6a7a9538534f7da1e4c303d2a318a728", "hex")
    # cipher= Present(key)
    # encrypted = cipher.encrypt(plain)
    # enc_1 = codecs.encode(encrypted, "hex")
    # print(enc_1)
    # "decryption"
    # decrypted = cipher.decrypt(encrypted)
    # dec_1 = codecs.encode(decrypted, "hex")
    # print(dec_1)

    # key = codecs.decode("00000000000000000000000000000000", "hex")
    # plain = codecs.decode(
    #     "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72", "hex"
    # )
    # auth_data = codecs.decode(
    #     "D609B1F056637A0D46DF998D88E52E00B2C2846512153524C0895E81108000F1", "hex"
    # )
    # iv = codecs.decode(
    #     "9313225df88406e555909c5aff5269aa6a7a9538534f7da1e4c303d2a318a728", "hex"
    # )
    # cipher, tag = gcm_encrypt(Present(key, iv, plain, auth_data))
    # encrypted = cipher.encrypt(plain)
    # enc_1 = codecs.encode(encrypted, "hex")
    # print(enc_1)
    # "decryption"
    # decrypted = cipher.decrypt(encrypted)
    # dec_1 = codecs.encode(decrypted, "hex")
    # print(dec_1)
