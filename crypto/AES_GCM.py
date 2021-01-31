from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
import os
import codecs


class Crypto:
    @staticmethod
    def encrypt_AES_GCM(msg, password, kdf_salt=None, nonce=None):
        """Used for encryption of msg"""

        kdf_salt = kdf_salt or os.urandom(16)
        nonce = nonce or os.urandom(16)

        # Encoding of message
        msg = msg.encode()
        secret_key = Crypto.generate_key(kdf_salt, password)
        aes_cipher = AES.new(secret_key, AES.MODE_GCM, nonce=nonce)
        ciphertext, auth_tag = aes_cipher.encrypt_and_digest(msg)

        return (kdf_salt, ciphertext, nonce, auth_tag)

    @staticmethod
    def decrypt_AES_GCM(encryptedMsg, password, kdf_salt=None, nonce=None):
        """Used for decryption of msg"""

        (stored_kdf_salt, ciphertext, stored_nonce, auth_tag) = encryptedMsg
        kdf_salt = kdf_salt or stored_kdf_salt
        nonce = nonce or stored_nonce

        secret_key = Crypto.generate_key(kdf_salt, password)
        aes_cipher = AES.new(secret_key, AES.MODE_GCM, nonce=nonce)
        plaintext = aes_cipher.decrypt_and_verify(ciphertext, auth_tag)

        # decoding byte data to normal string data
        plaintext = plaintext.decode("utf8")

        return plaintext

    @staticmethod
    def generate_key(kdf_salt, password, iterations=16384, r=8, p=1, buflen=32):
        """Generates the key that is used for encryption/decryption"""

        secret_key = scrypt(password, kdf_salt, N=iterations, r=r, p=p, key_len=buflen)
        return secret_key


def main():
    # http://www.ieee802.org/1/files/public/docs2011/bn-randall-test-vectors-0511-v1.pdf
    # k = "AD7A2BD03EAC835A6F620FDCB506B345"
    # k = codecs.decode(k, "hex")
    # p = ""
    # a = "D609B1F056637A0D46DF998D88E5222AB2C2846512153524C0895E8108000F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F30313233340001"
    # a = codecs.decode(a, "hex")
    # iv = "12153524C0895E81B2C28465"
    # iv = codecs.decode(iv, "hex")
    # c, t = gcm_encrypt(k, iv, "", a)
    # assert c == ""
    # assert t == "f09478a9b09007d06f46e9b6a1da25dd".decode("hex")

    password = "password123"
    msg = "password123"
    cipher = Crypto()
    encrypted = cipher.encrypt_AES_GCM(msg, password)
    # enc_1 = codecs.encode("encrypted", "utf-8")
    print(encrypted)

    "decryption"
    cipher = Crypto()
    decrypted = cipher.decrypt_AES_GCM(encrypted, msg)
    dec_1 = codecs.encode(decrypted, "utf-8")
    print(dec_1)


if __name__ == "__main__":
    main()