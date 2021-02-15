from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
import os
import codecs
from Crypto.Hash import SHA256


# salt = b"\x12\xcc\xd6E3\xc4\x80+\xd5\xaa\x9d\x8buq\xb7\x01"
# password = "hello"
# secret_key = scrypt(password, salt=salt, N=16384, r=16, p=1, key_len=16)
secret_key = b"Nn,S\xe7\xe6\xc5\xf1\xd4\x86\xe0'\xc6\x15Lu"


def encrypt_AES_GCM(info, nonce, secret_key):
    msg = info
    msg = msg.encode()
    nonce = b"\x8dj\x9c\x96\xcdf(O\xa6\xcfw\xd1\xa5\x8a\xc2\x99"

    cipher = AES.new(secret_key, AES.MODE_GCM, nonce=nonce)
    ciphertext = cipher.encrypt(msg)
    auth_tag = cipher.digest()
    return ciphertext, auth_tag


def decrypt_AES_GCM(ciphertext, secret_key, nonce):

    decipher = AES.new(secret_key, AES.MODE_GCM, nonce=nonce)
    plaintext = decipher.decrypt_and_verify(ciphertext, auth_tag)
    plaintext = plaintext.decode()

    " unpading"
    pad_index = plaintext.find(PAD)
    result = plaintext[:pad_index]
    return result


if __name__ == "__main__":

    "encryption"

    msg = "I love python"
    BLOCK_SIZE = 16
    PAD = "{"
    msg = msg + (BLOCK_SIZE - len(msg) % BLOCK_SIZE) * PAD
    nonce = b"\x8dj\x9c\x96\xcdf(O\xa6\xcfw\xd1\xa5\x8a\xc2\x99"
    cipher_text, auth_tag = encrypt_AES_GCM(msg, nonce=nonce, secret_key=secret_key)
    print("cipher_text_______", cipher_text)
    print("auth_tag______", auth_tag)
    print("cipher_text_len___", len(cipher_text))
    print("auth_tag_len___", len(auth_tag))

    "decryption"
    secret_key = b"Nn,S\xe7\xe6\xc5\xf1\xd4\x86\xe0'\xc6\x15Lu"
    plain_text = decrypt_AES_GCM(cipher_text, secret_key, nonce)
    print("plain_text___", plain_text)


# def generate_key(password, salt, key_len=16):
#     # password = password.encode()
#     secret_key = scrypt(password, salt, key_len=32, N=2 ** 17, r=16, p=1)
#     return secret_key


# def encrypt_AES_GCM(plaintext, password):

#     plaintext = codecs.encode("plaintext")
#     salt = os.urandom(16)
#     nonce = os.urandom(16)
#     block_size = 16
#     PAD = "{"
#     PAD = PAD.encode()
#     padding = lambda s: s + (block_size - len(s) % block_size) * PAD

#     # string can not be passed to generate key
#     secret_key = generate_key(password, salt)
#     aes_cipher = AES.new(secret_key, AES.MODE_GCM)
#     cipher_text = aes_cipher.encrypt(padding(plaintext))
#     auth_tag = aes_cipher.digest()

#     return cipher_text, auth_tag


# if __name__ == "__main__":
#     plaintext = "I love python"
#     password = "hello"
#     password = codecs.encode("password")
#     cipher_text = encrypt_AES_GCM(plaintext, password)
#     print(cipher_text)


# def decrypt(cipher_text, password):


# N – iterations count (affects memory and CPU usage), e.g. 16384 or 2048
# r – block size (affects memory and CPU usage), e.g. 8
# p – parallelism factor (threads to run in parallel - affects the memory, CPU usage), usually 1
# password– the input password (8-10 chars minimal length is recommended)
# salt – securely-generated random bytes (64 bits minimum, 128 bits recommended)
# derived-key-length - how many bytes to generate as output, e.g. 32 bytes (256 bits)


# from Crypto.Random import get_random_bytes
# from Crypto.Cipher import AES
# from Crypto.Protocol.KDF import scrypt
# import os
# import codecs


# class Crypto:
#     def __init__(self):
#         self.kdf_salt = None
#         self.nonce = None

#     def encrypt_AES_GCM(self, msg, password, kdf_salt=None, nonce=None):

#         kdf_salt = kdf_salt or os.urandom(16)
#         nonce = nonce or os.urandom(16)
#         self.kdf_salt = kdf_salt
#         self.nonce = nonce

#         # Encoding of message
#         msg = msg.encode()
#         secret_key = Crypto.generate_key(self, kdf_salt, password)
#         aes_cipher = AES.new(secret_key, AES.MODE_GCM)
#         ciphertext = aes_cipher.encrypt(msg)
#         auth_tag = aes_cipher.digest()

#         return (kdf_salt, ciphertext, nonce, auth_tag)

#     def decrypt_AES_GCM(self, encryptedMsg, password, kdf_salt=None, nonce=None):
#         (stored_kdf_salt, ciphertext, stored_nonce, auth_tag) = encryptedMsg
#         # kdf_salt = kdf_salt or stored_kdf_salt
#         # nonce = nonce or stored_nonce
#         kdf_salt = self.kdf_salt
#         nonce = self.nonce

#         secret_key = Crypto.generate_key(self, kdf_salt, password)
#         aes_cipher = AES.new(secret_key, AES.MODE_GCM, nonce=nonce)
#         plaintext = aes_cipher.decrypt_and_verify(ciphertext, auth_tag)

#         # decoding byte data to normal string data
#         plaintext = plaintext.decode("utf8")

#         return plaintext

#     def generate_key(self, kdf_salt, password, iterations=16384, r=8, p=1, buflen=32):
#         secret_key = scrypt(password, kdf_salt, N=iterations, r=r, p=p, key_len=buflen)
#         return secret_key


# if __name__ == "__main__":
#     msg = "password1234"
#     password = "password1234"
#     cipher = Crypto()
#     cipher_text = cipher.encrypt_AES_GCM(msg, password)
#     print("cipher_text____", cipher_text)
#     # print("auth_tag____", auth_tag)

#     "decryption"

#     plain_text = cipher.decrypt_AES_GCM(cipher_text, password)
#     dec_1 = codecs.encode(plain_text, "utf-8")

#     print("plaintext___", plain_text)
#     # dec_1 = codecs.encode(plain_text, "utf-8")
#     print(plain_text)


secret_key = b"Nn,S\xe7\xe6\xc5\xf1\xd4\x86\xe0'\xc6\x15Lu"
