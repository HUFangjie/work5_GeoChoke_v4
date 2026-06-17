import math, time
from typing import Dict, Any, List
import numpy as np, tenseal as ts
from crypto.base import CryptoBackend
from crypto.ckks_context_manager import CKKSContextManager
from crypto.ciphertext_payload import EncryptedUpdate, CiphertextChunk
from utils.validation import ensure_finite_array

class CKKSBackend(CryptoBackend):
    def __init__(self, context_manager:CKKSContextManager, profiles:Dict[str,Dict[str,Any]]):
        self.context_manager=context_manager; self.profiles=profiles; self.last_encryption_time=0.0
    def initialize_profiles(self): self.context_manager.initialize()
    def get_public_context(self, profile_id): return self.context_manager.public(profile_id)
    def encrypt_update(self, flat_update, profile_id):
        ensure_finite_array(flat_update, 'flat_update'); arr=np.asarray(flat_update,dtype=float); slot=self.context_manager.slot_count(profile_id)
        chunks=[]; t0=time.perf_counter()
        for idx,start in enumerate(range(0,len(arr),slot)):
            vals=arr[start:start+slot].tolist(); vec=ts.ckks_vector(self.get_public_context(profile_id), vals)
            chunks.append(CiphertextChunk(idx,len(vals),profile_id,len(arr),vec))
        self.last_encryption_time=time.perf_counter()-t0
        return EncryptedUpdate(chunks, profile_id, len(arr))
    def multiply_plain(self, encrypted_update, weight):
        if not np.isfinite(weight): raise ValueError('non-finite aggregation weight')
        return EncryptedUpdate([CiphertextChunk(c.chunk_index,c.valid_length,c.profile_id,c.total_dimension,c.payload*float(weight)) for c in encrypted_update.chunks], encrypted_update.profile_id, encrypted_update.total_dimension)
    def add_ciphertexts(self, ciphertexts):
        if not ciphertexts: raise ValueError('missing ciphertexts')
        pid=ciphertexts[0].profile_id; n=ciphertexts[0].block_count; dim=ciphertexts[0].total_dimension
        for ct in ciphertexts:
            if ct.profile_id!=pid: raise ValueError('profile mismatch during encrypted aggregation')
            if ct.block_count!=n or ct.total_dimension!=dim: raise ValueError('chunk mismatch during encrypted aggregation')
        out=[]
        for j in range(n):
            acc=ciphertexts[0].chunks[j].payload
            for ct in ciphertexts[1:]: acc=acc+ct.chunks[j].payload
            c0=ciphertexts[0].chunks[j]; out.append(CiphertextChunk(j,c0.valid_length,pid,dim,acc))
        return EncryptedUpdate(out,pid,dim)
    def serialize(self, ciphertext):
        return {"profile_id":ciphertext.profile_id,"total_dimension":ciphertext.total_dimension,"chunks":[{"chunk_index":c.chunk_index,"valid_length":c.valid_length,"profile_id":c.profile_id,"total_dimension":c.total_dimension,"payload":c.payload.serialize()} for c in ciphertext.chunks]}
    def deserialize(self, payload, profile_id):
        if payload['profile_id']!=profile_id: raise ValueError('profile mismatch on deserialize')
        ctx=self.get_public_context(profile_id); chunks=[]
        for c in payload['chunks']:
            chunks.append(CiphertextChunk(c['chunk_index'],c['valid_length'],profile_id,c['total_dimension'],ts.ckks_vector_from(ctx,c['payload'])))
        return EncryptedUpdate(chunks,profile_id,payload['total_dimension'])
