import numpy as np
class AuthorizedDecryptionService:
    def __init__(self, context_manager): self._context_manager=context_manager
    def decrypt_aggregate(self, aggregated_ciphertext, profile_id):
        if aggregated_ciphertext.profile_id!=profile_id: raise ValueError('profile mismatch for aggregate decryption')
        vals=[]
        for c in aggregated_ciphertext.chunks:
            dec=c.payload.decrypt(self._context_manager.secret(profile_id))[:c.valid_length]; vals.extend(dec)
        arr=np.array(vals,dtype=np.float64)
        if len(arr)!=aggregated_ciphertext.total_dimension: raise ValueError('decrypted dimension mismatch')
        return arr
