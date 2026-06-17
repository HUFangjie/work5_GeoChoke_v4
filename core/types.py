from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from crypto.ciphertext_payload import EncryptedUpdate
@dataclass
class ClientUpload:
    client_id:int; num_samples:int; profile_id:str; encrypted_update:EncryptedUpdate; metadata:Dict[str,Any]=field(default_factory=dict); plaintext_reference:Optional[Any]=None
