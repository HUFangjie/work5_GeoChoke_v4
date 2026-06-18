from dataclasses import dataclass
from typing import List, Any

@dataclass
class CiphertextChunk:
    chunk_index:int; valid_length:int; profile_id:str; total_dimension:int; payload:Any

@dataclass
class EncryptedUpdate:
    chunks:List[CiphertextChunk]; profile_id:str; total_dimension:int
    @property
    def block_count(self)->int: return len(self.chunks)
